from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='attempt1',
            executable='locate',
            name='locate_node',
        ),
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='v4l2_camera_node',
            arguments=['--ros-args', '-p', 'image_size:=640', '-p', 'image_size:=480'],
        )
    ])
