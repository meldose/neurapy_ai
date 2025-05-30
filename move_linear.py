import rclpy
from rclpy.node import Node
from moveit2 import MoveGroupInterface
from geometry_msgs.msg import Pose

class MoveLinearClient(Node):
    def __init__(self):
        super().__init__('move_linear_client')
        # “manipulator” is the name of your MoveIt2 group;
        # “world” is the planning frame.
        self.move_group = MoveGroupInterface(self, 'manipulator', 'world')

    def move_linear(self, dx: float, dy: float, dz: float):
        """
        Move the end‐effector along a straight line by (dx, dy, dz) in the planning frame.
        """
        # 1) Grab the current pose
        start_pose: Pose = self.move_group.get_current_pose().pose

        # 2) Build a single waypoint at the line end
        target = Pose()
        target.position.x = start_pose.position.x + dx
        target.position.y = start_pose.position.y + dy
        target.position.z = start_pose.position.z + dz
        target.orientation = start_pose.orientation
        waypoints = [target]

        # 3) Compute the Cartesian path
        (plan, fraction) = self.move_group.compute_cartesian_path(
            waypoints=waypoints,
            eef_step=0.01,        # 1 cm resolution
            jump_threshold=0.0    # disable jump checks
        )
        if fraction < 1.0:
            self.get_logger().warn(f'Only {fraction*100:.1f}% of the path was planned.')

        # 4) Execute the resulting trajectory
        self.move_group.execute(plan)

def main(args=None):
    rclpy.init(args=args)
    node = MoveLinearClient()
    # move 10 cm straight up
    node.move_linear(0.0, 0.0, 0.10)
    rclpy.spin_until_future_complete(node)  # spins until shutdown in execute()
