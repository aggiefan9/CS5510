import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Int32MultiArray

class ControlNode(Node):
    def __init__(self):
        super().__init__('control')
        self.sonar_subscription = self.create_subscription(
            Range, '/sonar', self.sonar_callback, 10
        )
        self.motor_publisher = self.create_publisher(
            Int32MultiArray, 'motor_control', 10
        )

    def sonar_callback(self, msg):
        dist = msg.range
        pub_msg = Int32MultiArray()

        if dist > 0.52:
            pub_msg.data = [50,50]
        elif dist < 0.48:
            pub_msg.data = [-50,-50]
        else:
            pub_msg.data = [0,0]

        self.motor_publisher.publish(pub_msg)

    def send_stop(self):
        msg = Int32MultiArray()
        msg.data = [0,0]
        self.motor_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    controller = ControlNode()

    try:
        rclpy.spin(controller)
    except:
        controller.send_stop()

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
