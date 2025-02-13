import rclpy
from rclpy.node import Node
from picamera2 import Picamera2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera')
        self.publisher = self.create_publisher(Image, 'image/rgb', 10)
        timer_period = 1/10
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.picam2 = Picamera2()
        self.picam2.start()
        self.bridge = CvBridge()

    def timer_callback(self):
        img = self.picam2.capture_array()
        self.publisher.publish(self.bridge.cv2_to_imgmsg(img,'bgr8'))

def main(args=None):
    rclpy.init(args=args)
    publisher = CameraPublisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
