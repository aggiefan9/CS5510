import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32MultiArray

class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape

class LocateNode(Node):
    def __init__(self):
        super().__init__('locate')
        self.subscription = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10
        )
        self.is_ball_publisher = self.create_publisher(Bool, 'is_ball', 10)
        self.ball_info_publisher = self.create_publisher(
            Int32MultiArray, 'ball_info', 10)
        self.cv_bridge = CvBridge()

        # Create an instance of the ShapeDetector class
        self.shape_detector = ShapeDetector()

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Your ball detection and position determination logic here
        is_ball, ball_info = self.detect_ball(cv_image)

        # Publish results to topics
        is_ball_msg = Bool()
        is_ball_msg.data = is_ball
        self.is_ball_publisher.publish(is_ball_msg)

        ball_info_msg = Int32MultiArray(data=ball_info)
        self.ball_info_publisher.publish(ball_info_msg)

    def detect_ball(self, cv_image):
        # Camera intrinsic parameters from calibration
        mtx = np.array([[675.93879896, 0.0, 991.0434498],
            [0.0, 678.36517214, 555.23630622],
            [0.0, 0.0, 1.0]])
        dist = np.array([[-0.02077507, 0.0199203, 0.00048656, -0.00125545, -0.0201533]])

        # mtx = np.array([[1.51424068e+03, 0.0, 3.19650358e+02],
        #     [0.0, 1.55503723e+03, 2.39353296e+02],
        #     [0.0, 0.0, 1.0]])
        # dist = np.array([[ 2.21471857, -6.82352846e+02,  3.00306827e-02,  5.24950588e-02, -3.24969429]])

        # transformation_matrix = np.array([
        #     [-9.96183603e-01, 7.49769589e-02, 4.46842867e-02, 3.12709637e+00],
        #     [-8.27532785e-02,-9.74104116e-01,-2.10411657e-01, 4.19386410e-01],
        #     [ 2.77511215e-02,-2.13306414e-01, 9.76591137e-01, 7.50657062e+01],
        #     [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        # ])

        # Undistort the image using calibration parameters
        undistorted_image = cv2.undistort(cv_image, mtx, dist, None, mtx)

        # Convert the undistorted image to HSV for better color segmentation
        hsv_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)

        # Define color ranges for the balls in HSV format
        # These values are just placeholders; adjust them based on your ball colors
        lower_color_range = np.array([0, 100, 100])
        upper_color_range = np.array([10, 255, 255])

        # Create a mask using the color range
        color_mask = cv2.inRange(hsv_image, lower_color_range, upper_color_range)

        # Find contours in the color mask
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_ball = False
        ball_info = [0, 0, 0]  # Default coordinates and distance if the ball is not found

        if contours:
            # Assume the largest contour is the ball
            largest_contour = max(contours, key=cv2.contourArea)

            # Use the ShapeDetector to detect if the contour is a circle
            shape = self.shape_detector.detect(largest_contour)

            # if shape == "circle":
            if shape:
                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Calculate the center of the bounding box
                center_x = x + w // 2
                center_y = y + h // 2

                # Convert pixel coordinates to real-world coordinates
                pixel_coordinates = np.array(
                    [[center_x, center_y]], dtype=np.float32).reshape(-1, 1, 2)
                undistorted_coordinates = cv2.undistortPoints(
                    pixel_coordinates, mtx, dist, P=mtx)

                # Extract undistorted coordinates
                center_x_true, center_y_true = undistorted_coordinates[0][0]
                center_x_true = int(center_x_true)
                center_y_true = int(center_y_true)

                # Distance to the ball (simple conversion)
                distance_to_ball = int(np.sqrt(
                    (center_x_true - mtx[0, 2])**2 + (center_y_true - mtx[1, 2])**2
                ))

                # Transform pixel coordinates to 3D coordinates
                # h_pixel_coordinates = np.array([center_x_true, center_y_true, 1, 1])
                # h_robot_coordinates = np.dot(transformation_matrix, h_pixel_coordinates)
                # robot_coordinates = h_robot_coordinates[:3]

                is_ball = True
                ball_info = [center_x_true, center_y_true, distance_to_ball]
                # ball_info = [int(robot_coordinates[0]), int(robot_coordinates[1]), int(robot_coordinates[2])]

        return is_ball, ball_info

def main(args=None):
    rclpy.init(args=args)

    locator = LocateNode()

    rclpy.spin(locator)

    locator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
