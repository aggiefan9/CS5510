import os
import cv2
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class ImageNode:
    def __init__(self):
        self.node = rclpy.create_node('image')
        self.image_sub = self.node.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.cv_bridge = CvBridge()
        self.img_name = 1
        # Create 'img' folder if it doesn't exist
        self.img_folder = os.path.join(os.getcwd(), 'img')
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.node.get_logger().error(f"Error converting image: {str(e)}")
            return

        # Save the image to the 'img' folder
        image_filename = f"image_{self.img_name}.jpg"
        image_path = os.path.join(self.img_folder, image_filename)
        cv2.imwrite(image_path, cv_image)
        self.node.get_logger().info(f"Image saved: {image_path}")
        self.img_name += 1

def main(args=None):
    rclpy.init(args=args)
    image_node = ImageNode()
    rclpy.spin(image_node.node)
    image_node.node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
