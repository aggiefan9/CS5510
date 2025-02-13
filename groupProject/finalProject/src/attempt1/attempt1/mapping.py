import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Range, Image
from std_msgs.msg import Bool, Int32MultiArray

class MappingNode(Node):
    def __init__(self):
        super().__init__('mapping_node')
        self.subscription_image = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        self.subscription_sonar = self.create_subscription(
            Range,
            '/sonar',
            self.sonar_callback,
            10
        )
        self.subscription_is_ball = self.create_subscription(
            Bool,
            '/is_ball',
            self.is_ball_callback,
            10
        )
        self.subscription_ball_info = self.create_subscription(
            Int32MultiArray,
            '/ball_info',
            self.ball_info_callback,
            10
        )
        self.publisher_occupancy_grid = self.create_publisher(
            Int32MultiArray,
            '/occupancy_grid',
            10
        )

        self.occupancy_grid = np.zeros((100, 100), dtype=np.uint8)  # Assuming a 100x100 grid for simplicity
        self.latest_sonar_data = None
        self.latest_ball_info = None

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Your obstacle detection and occupancy grid update logic here
        self.update_occupancy_grid(cv_image)
        self.publish_occupancy_grid()

    def update_occupancy_grid(self, cv_image):
        # Convert the image to grayscale for simplicity (you can use more advanced techniques)
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to identify obstacles (example: simple thresholding)
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Update occupancy grid based on detected obstacles from camera data
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Convert obstacle coordinates to grid coordinates
            grid_x = int(x / cv_image.shape[1] * 100)  # Assuming a 100x100 grid
            grid_y = int(y / cv_image.shape[0] * 100)

            # Mark the corresponding cell in the grid as occupied
            self.occupancy_grid[grid_y, grid_x] = 255

        # Update occupancy grid based on sonar data
        if self.latest_sonar_data:
            sonar_range = self.latest_sonar_data.range
            sonar_angle = self.latest_sonar_data.field_of_view / 2.0

            # Convert sonar angle range to pixel range in the image
            pixel_range_start = int(
                (0.5 - sonar_angle / 40) * cv_image.shape[1])
            pixel_range_end = int((0.5 + sonar_angle / 40) * cv_image.shape[1])

            # Mark cells within the sonar range as occupied
            for pixel_x in range(pixel_range_start, pixel_range_end):
                grid_x = int(pixel_x / cv_image.shape[1] * 100)
                self.occupancy_grid[:, grid_x] = 255

    def sonar_callback(self, msg):
        # Store the latest sonar data
        self.latest_sonar_data = msg

    def is_ball_callback(self, msg):
        # Update occupancy grid based on whether a ball is detected
        if msg.data:
            # Ball detected, update the corresponding cells in the grid
            # Use the information from the /ball_info topic to update the grid
            # (Assuming /ball_info contains [center_x, center_y, distance_to_ball])
            center_x, center_y, _ = self.latest_ball_info
            grid_x = int(center_x / msg.width * 100)  # Assuming a 100x100 grid
            grid_y = int(center_y / msg.height * 100)
            
            # Mark the corresponding cell in the grid as occupied
            self.occupancy_grid[grid_y, grid_x] = 255

    def ball_info_callback(self, msg):
        # Store the latest ball information
        self.latest_ball_info = msg.data

    def publish_occupancy_grid(self):
        # Convert the occupancy grid to Int8MultiArray message
        occupancy_grid_msg = Int32MultiArray()
        occupancy_grid_msg.data = self.occupancy_grid.flatten().tolist()

        # Publish the occupancy grid
        self.publisher_occupancy_grid.publish(occupancy_grid_msg)


def main(args=None):
    rclpy.init(args=args)

    mapping_node = MappingNode()

    rclpy.spin(mapping_node)

    mapping_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
