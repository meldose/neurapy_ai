#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# Import the action definition
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class CartesianToJointClient(Node):
    def __init__(self):
        super().__init__('cartesian_to_joint_client')

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        self.joint_names = [
            'joint1', 'joint2', 'joint3',
            'joint4', 'joint5', 'joint6',
            'joint7',
        ]

        self.get_logger().info("Waiting for action server '/joint_trajectory_position_controller/follow_joint_trajectory'...")
        self._action_client.wait_for_server()
        self.get_logger().info("Action server is available. Ready to send goals.")

        self.send_test_trajectory()

    def send_test_trajectory(self):
        """
        Build a simple two‐point joint trajectory and send it as a single goal.
        """


        goal_msg = FollowJointTrajectory.Goal()

        goal_msg.trajectory.joint_names = self.joint_names


        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()


        pt1 = JointTrajectoryPoint()

        pt1.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pt1.velocities = [0.0] * len(self.joint_names)
        pt1.accelerations = [0.0] * len(self.joint_names)
        pt1.time_from_start = Duration(sec=1, nanosec=0)


        pt2 = JointTrajectoryPoint()

        pt2.positions = [0.5, -0.3, 0.2, -0.5, 0.1, 0.0, -0.2]
        pt2.velocities = [0.0] * len(self.joint_names)
        pt2.accelerations = [0.0] * len(self.joint_names)
        pt2.time_from_start = Duration(sec=4, nanosec=0)

        goal_msg.trajectory.points = [pt1, pt2]

        self.get_logger().info("Sending joint‐space trajectory goal to the controller...")
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Called when the action server has accepted or rejected our goal.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by the action server.")
            return

        self.get_logger().info("Goal accepted by the action server. Waiting for result…")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """
        Called whenever the action server sends feedback. You can inspect
        feedback_msg.feedback to see real‐time trajectory execution status.
        """
        fb = feedback_msg.feedback

        self.get_logger().info(
            f"Feedback: actual positions = {['{:.2f}'.format(p) for p in fb.actual.positions]}"
        )

    def get_result_callback(self, future):
        """
        Called when the action server finishes executing the trajectory (either success or failure).
        """
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info(" Trajectory execution succeeded (error_code == 0).")
        else:
            self.get_logger().error(
                f" Trajectory execution failed! error_code = {result.error_code}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = CartesianToJointClient()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
