import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Int32MultiArray

class MoveNode(Node):
    def __init__(self):
        super().__init__('move')
        self.sonar_subscription = self.create_subscription(
            Range, '/sonar', self.sonar_callback, 10
        )
        self.ball_info_subscribtion = self.create_subscription(
            Int32MultiArray, '/ball_info', self.ball_callback, 10
        )
        self.motor_publisher = self.create_publisher(
            Int32MultiArray, 'motor_control', 10
        )
        self.dist = -1
        self.ball_data = [0,0,0]

    def sonar_callback(self, msg):
        self.dist = msg.range
        self.move_to_ball()

    def ball_callback(self, msg):
        self.ball_data = msg.data
        self.move_to_ball()

    def move_to_ball(self):
        msg = Int32MultiArray()

        if self.dist != -1 and self.ball_data[2] != 0:
            if self.ball_data[0] > 420:
                msg.data = [50,0]
            elif self.ball_data[0] < 270:
                msg.data = [0,50]
            else:
                if self.dist > 0.05:
                    msg.data = [50,50]
                else:
                    msg.data = [0,0]
        else:
            msg.data = [0,0]

        self.motor_publisher.publish(msg)

    def send_stop(self):
        msg = Int32MultiArray(data=[0,0])
        self.motor_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    move_node = MoveNode()
    rclpy.spin(move_node)
    move_node.send_stop()
    move_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
