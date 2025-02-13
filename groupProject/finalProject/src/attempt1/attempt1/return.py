import cv2
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Int32MultiArray

class MoveNode(Node):
    def __init__(self):
        super().__init__('move')
        self.sonar_subscription = self.create_subscription(
            Range, '/sonar', self.sonar_callback, 10
        )
        self.goal_info_subscribtion = self.create_subscription(
            Int32MultiArray, '/goal_info', self.goal_callback, 10
        )
        self.motor_publisher = self.create_publisher(
            Int32MultiArray, 'motor_control', 10
        )
        self.dist = -1
        self.goal_data = [0,0,0]
        self.seen_goal = False
        self.goal_dist = -1
        self.last_goal_loc = 0

    def sonar_callback(self, msg):
        self.dist = msg.range
        self.move_to_goal()

    def goal_callback(self, msg):
        self.goal_data = msg.data
        self.move_to_goal()

    def move_to_goal(self):
        msg = Int32MultiArray()
        self.goal_dist = self.dist
        self.seen_goal = True
        self.last_goal_loc = self.goal_data[0]
        
        if self.dist != -1 and self.goal_data[2] != 0: ## If we see the goal
            if self.last_goal_loc > 450: # The goal is to the left of us
                msg.data = [50,0]
            elif self.last_goal_loc < 240: # The goal is to the right of us
                msg.data = [0,50]
            else: # The goal is centered
                if self.dist > 0.05:
                    msg.data = [50,50]
                else:
                    msg.data = [0,0]
        elif self.seen_goal: # If we don't see the goal, but we have before
            if self.last_goal_loc > 450: # If we saw it to the left of us
                msg.data = [0,50]
            elif self.last_goal_loc < 240: # If we saw it to the right of us
                msg.data = [50,0]
            else: # If we saw it right in front of us
                if self.dist > 0.05: # Assume it's just below the camera, and move to touch it
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
