import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import numpy as np

class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planning_node')
        self.subscription_ball_info = self.create_subscription(
            Int32MultiArray,
            '/ball_info',
            self.ball_info_callback,
            10
        )

        self.subscription_occupancy_grid = self.create_subscription(
            Int32MultiArray,
            '/occupancy_grid',
            self.occupancy_grid_callback,
            10
        )

        self.publisher_motor_control = self.create_publisher(
            Int32MultiArray,
            '/motor_control',
            10
        )

        self.occupancy_grid = None
        self.target_position = None
        self.robot_position = None

    def ball_info_callback(self, msg):
        # Extract ball information from the message
        center_x, center_y, _ = msg.data

        # Set the target position based on the ball's center coordinates
        self.target_position = (center_x, center_y)

        # Plan and execute the path
        self.plan_and_execute_path()

    def occupancy_grid_callback(self, msg):
        # Extract occupancy grid data from the message
        self.occupancy_grid = np.array(msg.data).reshape((100, 100))

        # Plan and execute the path
        self.plan_and_execute_path()

    def plan_and_execute_path(self):
        if self.occupancy_grid is not None and self.target_position is not None:
            # Plan a path based on the occupancy grid and target position
            path = self.plan_path(self.occupancy_grid, self.target_position)

            # Execute the planned path
            self.execute_path(path)

    def plan_path(self, occupancy_grid, target_position):
        # For simplicity, plan a path that avoids occupied cells in the occupancy grid
        # You might want to use a more sophisticated path planning algorithm in a real-world scenario

        # Example: A* or Dijkstra's algorithm
        # Placeholder implementation, replace with your actual path planning logic

        # In this example, we just find a path from the robot's position (assumed center of the grid)
        # to the target position, avoiding occupied cells in the occupancy grid.

        # Assuming a 2D grid with a size of 100x100
        grid_size = 100
        self.robot_position = (grid_size // 2, grid_size // 2)

        # Create a simple path by moving towards the target position while avoiding occupied cells
        path = [self.robot_position]

        current_position = self.robot_position

        while current_position != target_position:
            x, y = current_position
            target_x, target_y = target_position

            # Move towards the target position while avoiding occupied cells
            if x < target_x and occupancy_grid[y, x + 1] == 0:
                current_position = (x + 1, y)
            elif x > target_x and occupancy_grid[y, x - 1] == 0:
                current_position = (x - 1, y)
            elif y < target_y and occupancy_grid[y + 1, x] == 0:
                current_position = (x, y + 1)
            elif y > target_y and occupancy_grid[y - 1, x] == 0:
                current_position = (x, y - 1)

            path.append(current_position)

        return path

    def execute_path(self, path):
        # Execute the planned path by publishing motor control commands
        for position in path:
            # Convert path coordinates to motor control commands (for simplicity, assuming a differential drive robot)
            delta_x = position[0] - self.robot_position[0]
            delta_y = position[1] - self.robot_position[1]

            # Placeholder conversion to motor control commands
            left_motor_command = int(delta_x * 2)
            right_motor_command = int(delta_y * 2)

            # Publish motor control commands
            motor_control_msg = Int32MultiArray()
            motor_control_msg.data = [left_motor_command, right_motor_command]
            self.publisher_motor_control.publish(motor_control_msg)

            # Update the robot's position
            self.robot_position = position

        # Stop the car when the path is completed
        stop_command_msg = Int32MultiArray()
        stop_command_msg.data = [0, 0]
        self.publisher_motor_control.publish(stop_command_msg)

def main(args=None):
    rclpy.init(args=args)

    motion_planning_node = MotionPlanningNode()

    rclpy.spin(motion_planning_node)

    motion_planning_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
