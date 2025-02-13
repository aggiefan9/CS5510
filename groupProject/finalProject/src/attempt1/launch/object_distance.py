from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='attempt1',
            executable='sonar',
            name='sonar_node',
        ),
        Node(
            package='attempt1',
            executable='motors',
            name='motors_node',
        ),
        Node(
            package='attempt1',
            executable='control',
            name='control_node',
        )
    ])
