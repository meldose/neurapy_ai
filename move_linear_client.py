import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# MoveIt2 Python binding
from moveit2.move_group_interface import MoveGroupInterface

from control_msgs.action import FollowJointTrajectory


class MoveLinearClient(Node):
    def __init__(self):
        super().__init__('move_linear_client')

        # 1) Create MoveGroupInterface for your arm
        #
        #    - group_name must match the MoveIt “planning group” (e.g. “arm”)
        #    - base_frame and ee_frame must match your URDF
        self.move_group = MoveGroupInterface(
            node=self,
            group_name='arm',
            base_frame='base_link',
            ee_frame='ee_link',
        )

        # 2) Make an ActionClient for your position controller
        self._traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        # 3) Wait until the controller’s action server is up
        if not self._traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('FollowJointTrajectory action server not available!')
            rclpy.shutdown()
            return

    def move_linear(self, dx: float, dy: float, dz: float):
        # A) Get the current end-effector pose
        start = self.move_group.get_current_pose().pose

        # B) Build the target pose
        target = start.__class__()  # Pose()
        target.position.x = start.position.x + dx
        target.position.y = start.position.y + dy
        target.position.z = start.position.z + dz
        target.orientation = start.orientation

        # C) Plan a cartesian path
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints=[target],
            eef_step=0.01,       # 1 cm segments
            jump_threshold=0.0,  # disable jump checks
        )
        if fraction < 1.0:
            self.get_logger().warn(f'Only {fraction*100:.1f}% of the path could be planned.')

        # D) Wrap the RobotTrajectory in a FollowJointTrajectory goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = plan.joint_trajectory

        # E) Send the goal
        send_goal_future = self._traj_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal rejected by controller')
            return

        # F) Wait for result
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        if result.error_code != result.SUCCESSFUL:
            self.get_logger().error(f'Execution failed: code {result.error_code}')
        else:
            self.get_logger().info('Motion executed successfully!')


def main(args=None):
    rclpy.init(args=args)
    node = MoveLinearClient()
    # for example, move straight up 10 cm:
    node.move_linear(0.0, 0.0, 0.10)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
