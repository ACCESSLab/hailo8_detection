**Last HailoRT version checked - 4.14.0**

## ROS2 Package for the HailoRT python3.10 binding, implementing yolov8

### How to run
```shell
ros2 launch hailo8_detection hailo8_detection.launch.py
```

#### Works with this [ROS2 USB Camera package](https://github.com/ros-drivers/usb_cam/tree/ros2). Make sure that config/params.yaml and config/camera_info.yaml match the hailo8_detection.launch.py params 

#### Download yolov8s.hef and place in models directory

```shell
bash scripts/get_hefs_and_video.sh
```

### Input

| Name       | Type                | Description     |
| ---------- | ------------------- | --------------- |
| `img_raw` | `sensor_msgs/Image` | input image |

### Output

| Name          | Type                                               | Description                                        |
| ------------- | -------------------------------------------------- | -------------------------------------------------- |
| `det_img`   | `sensor_msgs/Image`                                | Output image with bounding boxes |

### Node Parameters

| Name                    | Type   | Default Value | Description                                                        |
| ----------------------- | ------ | ------------- | ------------------------------------------------------------------ |
| `image_raw`             | string | "image_raw"            | Input image topic                                  |
| `image_width`             | double | 1920.0            | Input image width                                  |
| `image_height`             | double | 1080.0            | Input image height                                  |
| `arch`             | string | "yolo_v8"            | The model architecture                                  |
| `draw_bbox`             | bool | "True"            | Draw the bounding boxes on the image T/                               |
| `norm_img`             | bool | "True"            | Normalize the input image                                  |


**NOTE**: This example was only tested with the yolov8s.hef model.

### Requirements / Tested with

Hailo8 M.2

Python 3.10.6

ROS2 Humble

hailort 4.14

hailort-driver 4.14

tappas 3.25.0

OpenCV 4.2.X


