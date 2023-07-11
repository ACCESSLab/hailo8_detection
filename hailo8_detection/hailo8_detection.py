#!/usr/bin/env python3

import numpy as np
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams, FormatType)
from zenlog import log
from hailo_model_zoo.core.postprocessing.detection.nanodet import NanoDetPostProc
import cv2
import rclpy
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

kwargs = {}
kwargs['device_pre_post_layers'] = None

class yolov8_class():
    def __init__(self):
        self.strides = [32,16,8]
        self.regression_length = 15
        self.scale_factors = [0, 0]
        self.device_pre_post_layers = device_pre_post_layers()
class device_pre_post_layers():
    def __init__(self):
        self.sigmoid = True


class Hailo8_Detection(Node):

    def __init__(self,infer_pipeline, network_group,network_group_params, 
                 input_vstream_info, m_height, m_width):
        super().__init__('Hailo8_Detection')
        sensor_qos = qos_profile_sensor_data
        sensor_qos.reliability = QoSReliabilityPolicy.RELIABLE
        sensor_qos.depth=100
        self.subscription = self.create_subscription(Image, 'image_raw', self.image_callback, qos_profile=sensor_qos)
        self.bridge = CvBridge()
        self.img_w = self.declare_parameter('image_width', 1920.0).value
        self.img_h = self.declare_parameter('image_height', 1080.0).value
        self.draw_bbox = self.declare_parameter('draw_bbox', True).value
        arch = self.declare_parameter('arch', 'yolo_v8').value
        self.norm_img = self.declare_parameter('norm_img', True).value
        self.publisher = self.create_publisher(Image , "det_image", qos_profile=sensor_qos)
        self.model_w, self.model_h = m_width, m_height
        self.scaled_h_min = (self.img_w - self.img_h)/ self.img_w / 2.0
        self.scaled_h_max = 1.0 - self.scaled_h_min
        self.infer_pipeline = infer_pipeline
        self.network_group = network_group
        self.input_vstream_info = input_vstream_info
        self.network_group_params = network_group_params
        arch_dict = {'yolo_v8': {}}
        self.num_of_classes = 80
        self.func_dict = {'nanodet_v8': self.postproc_yolov8}

        self.anchors = {}
        self.meta_arch = ''
        arch_list = arch_dict.keys()
        if arch in arch_list:
            self.anchors = arch_dict[arch]
            if arch == 'yolo_v8':
                self.meta_arch = 'nanodet_v8'
   
    def image_callback(self, msg):
        now = self.get_clock().now()
        print(now)
        timestamp = msg.header
        image = self.bridge.imgmsg_to_cv2(msg,'rgb8')
        img_resized = self.resize_image(image,self.model_w,self.model_h)
        if self.norm_img == True:
            cv2.normalize(img_resized, img_resized, 0, 255, norm_type=cv2.NORM_MINMAX)
        input_data = {self.input_vstream_info.name: np.expand_dims(img_resized, axis=0).astype(np.float32)}
        with self.network_group.activate(self.network_group_params):
          raw_detections = self.infer_pipeline.infer(input_data)

        results = self.func_dict[self.meta_arch](self.model_h, self.model_w, self.anchors, self.meta_arch, int(self.num_of_classes), raw_detections)
        drawn_image, bbox, labels, score = self.post_process(image,results,self.img_w,self.img_h)
        if self.draw_bbox == True:
            ros_msg = self.bridge.cv2_to_imgmsg(drawn_image,'bgr8')
            self.publisher.publish(ros_msg)
        else:
            pass
            ##bbox arraymsg
        
        print((self.get_clock().now().nanoseconds- now.nanoseconds)/1e6)

    def resize_image(self,img,x,y):
        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape)>2 else 1
        size=(x,y)
        
        if h == w: 
            return cv2.resize(img, size, cv2.INTER_AREA)
        dif = h if h > w else w
        interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC
        x_pos = (dif - w)//2
        y_pos = (dif - h)//2

        if len(img.shape) == 2:
            mask = np.zeros((dif, dif), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
        else:
            mask = np.zeros((dif, dif, c), dtype=img.dtype)
            mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

        return cv2.resize(mask, size, interpolation)
  
    def postproc_yolov8(self,height, width, anchors, meta_arch, num_of_classes, raw_detections):
        raw_detections_keys = list(raw_detections.keys())
        raw_detections_keys.sort()
        yolov8_cls = yolov8_class()
        
        post_proc = NanoDetPostProc(img_dims=(height,width),
                                    anchors=yolov8_cls, 
                                    meta_arch=meta_arch, 
                                    classes=num_of_classes,
                                    nms_iou_thresh=0.7,
                                    score_threshold=0.001,
                                    **kwargs)
        
        layer_from_shape: dict = {raw_detections[key].shape:key for key in raw_detections_keys}

        detections = [raw_detections[layer_from_shape[1, 20, 20, 64]],
                        raw_detections[layer_from_shape[1, 20, 20, 80]],
                        raw_detections[layer_from_shape[1, 40, 40, 64]],
                        raw_detections[layer_from_shape[1, 40, 40, 80]],
                        raw_detections[layer_from_shape[1, 80, 80, 64]],
                        raw_detections[layer_from_shape[1, 80, 80, 80]]]    
      
        return post_proc.postprocessing(detections, device_pre_post_layers=yolov8_cls.device_pre_post_layers)
  
    def remap_height(self,value):
      # remap the height range since its referring to a letterbox image. assumption w>h like 4:3,16:9
      # (img_width - img_height) / img_width / 2 
      # is the amount of padding on the top and bottom of the image
      original_min, original_max  = self.scaled_h_min, self.scaled_h_max
      target_min, target_max = 0.0, 1.0
      remapped_value = (value - original_min) / (original_max - original_min) * (target_max - target_min) + target_min
      return remapped_value
    
    def post_process(self,image,detections, width, height, min_score=0.45, scale_factor=1):
        COLORS = np.random.randint(0, 255, size=(100, 2), dtype=np.uint8)
        boxes = np.array(detections['detection_boxes'])[0]
        classes = np.array(detections['detection_classes'])[0].astype(int)
        scores = np.array(detections['detection_scores'])[0]
        scaled_box = []
        for idx in range(np.array(detections['num_detections'])[0]):
          if scores[idx] >= min_score:
            
            scaled_box = [x*width if i%2 else self.remap_height(x)*height for i,x in enumerate(boxes[idx])]
            ymin, xmin, ymax, xmax = scaled_box
            if self.draw_bbox == True:
              color = tuple(int(c) for c in COLORS[classes[idx]])
              image = cv2.rectangle(image,(int(xmin),int(ymin)),
                                          (int(xmax),int(ymax)),
                                          color,thickness=2)
        return image, scaled_box, classes, scores


def main(args=None):
  
    from ament_index_python.packages import get_package_share_directory
    get_package_share_directory= get_package_share_directory('hailo8_detection')
    model_path= str(get_package_share_directory)+"/yolov8s.hef"
    devices = Device.scan()
    hef = HEF(model_path)

    # inputs = hef.get_input_vstream_infos()
    # outputs = hef.get_output_vstream_infos()
    with VDevice(device_ids=devices) as target:
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        # [log.info('Input  layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in inputs]
        # [log.info('Output layer: {} {}'.format(layer_info.name, layer_info.shape)) for layer_info in outputs]
        m_height, m_width, _ = hef.get_input_vstream_infos()[0].shape
        input_vstream_info = hef.get_input_vstream_infos()[0]
        input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            
            rclpy.init(args=args)
            hailo8_detection = Hailo8_Detection(infer_pipeline, network_group,
                                                network_group_params, input_vstream_info,
                                                m_height, m_width)

            rclpy.spin(hailo8_detection)
            hailo8_detection.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

