import launch
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    launch_args = []
    def add_launch_arg(name: str, default_value=None):
        launch_args.append(DeclareLaunchArgument(name, default_value=default_value))
    add_launch_arg("input_image", "image_raw")
    add_launch_arg("image_width", "1920.0")
    add_launch_arg("image_height", "1080.0")
    add_launch_arg("arch", "yolo_v8")
    add_launch_arg("draw_bbox", "True")
    add_launch_arg("norm_img", "True")
    nodes = [
        Node(
            package='hailo8_detection',
            executable='hailo8_detection',
            remappings =[
              ("image_raw", LaunchConfiguration("input_image"))
            ],
            parameters=[{
                "image_width": LaunchConfiguration("image_width"),
                "image_height": LaunchConfiguration("image_height"),
                "draw_bbox": LaunchConfiguration("draw_bbox"),
                "arch": LaunchConfiguration("arch"),
                "norm_img": LaunchConfiguration("norm_img"),
            }],
        ),
    ] 

    return launch.LaunchDescription(
        launch_args + nodes
    )