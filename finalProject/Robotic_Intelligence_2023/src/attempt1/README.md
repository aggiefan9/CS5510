To run the camera module, we just used the v4l2 camera node.
It can be run with the following command:

`ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[640,480]"`

The project only needs the dependencies for Taylor's ROS2 setup guide

To run everything. Run the following commands from the src folder:
1. `colcon build`
2. `. install/setup.zsh`
3. `cp -r attempt1/launch/ install/attempt1/share/attempt1/launch/`
4. `ros2 launch attempt1 launch.py`

There are 5 main nodes that our implementation uses. Their description is below:

1. Camera: Though there is a camera node in the code repository, we could not get that to function properly, so we are instead relying on the v4l2_camera_node that is implemented in the ROS2 standard library. This publishes an array form of an image taken by the camera to the /image_raw topic.
2. Sonar: Patterned after the implementation provided by Taylor Anderson this node uses GPIO to publish the distance to the nearest object to the /sonar topic.
3. Motor Control: Patterned after the implementation provided by Taylor Anderson, this node subscribes to the /motor_control and /servo_control topics, and utilizes the Car class also implemented by Taylor Anderson to convert the messages received into commands that are sent to the motors on the robot.
4. Locator: This node subscribes to the /image_raw topic, and uses OpenCV to perform a color mask on the image, then detects contours and publishes the x and y coordinates (alongside a distance estimate) of the largest shape (assumed to be the ball or goal respectively) to the /ball_info (or /goal_info) topic, as well as a boolean /is_ball flag representing the presence of the ball in the image.
5. Control Policy: This node subscribes to the /sonar and /ball\_info topics, and publishes the calculated motion commands to the /motor_control topic in order to center the ball in the frame of reference, and move towards it, stopping when it has reached it before attempting to find and move to the goal.

There are also nodes for mapping and planning in the repository. Once again, functionality of these nodes could not be achieved before the deadline, resulting in the project scope re-evaluation.

The code used for camera calibration is also present, in the calibrate.py and save.py files. The save.py node is a ROS node designed to be run in tandem with a camera node, and all it does is save images to an img folder.

Then calibrate.py uses these images (idealy taken of a grid for calibration) to calibrate the camera and get the mtx, dist, rvec, and tvec matrices used to undistort the images for image detection.