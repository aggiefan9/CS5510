To run the camera module, we just used the v4l2 camera node.
It can be run with the following command:

`ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[640,480]"`

The project only needs the dependencies for Taylor's ROS2 setup guide

To run everything. Run the following commands from the src folder:
1. `colcon build`
2. `. install/setup.zsh`
3. `cp -r attempt1/launch/ install/attempt1/share/attempt1/launch/`
4. `ros2 launch attempt1 launch.py`