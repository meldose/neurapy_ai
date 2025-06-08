#!/usr/bin/env python3

import rclpy # imported rclpy moudule
from rclpy.node import Node # imported Node 
from rclpy.action import ActionClient # imported Action client 
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint # imported JointTrajectory
from builtin_interfaces.msg import Duration # imported Duration
from sensor_msgs.msg import JointState #imported JOinstate
from std_msgs.msg import Bool # imported Bool
from geometry_msgs.msg import PoseArray, Pose #imported Pose

# created class for Plan MotionLinearVia Points
class PlanMotionLinearViaPoints(Node):
    def __init__(self):
        super().__init__('plan_motion_linear_via_points')

        # Create an ActionClient for FollowJointTrajectory
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        # Wait once for the action server to come up
        self.get_logger().info("Waiting for FollowJointTrajectory action server...")
        if not self._client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Action server not available after 10 seconds. Shutting down.")
            rclpy.shutdown()
            return
        self.get_logger().info("Action server is available.")

        # Subscribe to Cartesian waypoints topic
        self.create_subscription(
            PoseArray,
            '/cartesian_waypoints',
            self.move_linear_via_points,
            10
        )

        #  Publisher for success/failure result
        self.pub_result = self.create_publisher(Bool, '/mlp_result', 10)

        # Robot’s joint names (modify to match your robot)
        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
        ]

        # Placeholder for the result future
        self._result_future = None

# created function for ik solver
    def _ik_solver(self, pose: Pose) -> list:
        """
        Stub IK solver. Replace with a real IK implementation.
        Given a Pose, returns a list of joint positions [q1, q2, ..., q7].
        If IK fails, return None or raise an exception.
        """
        self.get_logger().warn("IK solver not implemented; returning zero positions.")
        return [0.0] * len(self.joint_names)

# created function for move_linear
    def move_linear_via_points(self, msg: PoseArray):
        """
        Callback executed whenever a PoseArray arrives on '/cartesian_waypoints'.
        Builds a single multi-point joint trajectory from all poses and sends it
        as one FollowJointTrajectory goal. Publishes a Bool on '/mlp_result'.
        """
        # Number of waypoints in PoseArray
        num_points = len(msg.poses)
        if num_points == 0:
            self.get_logger().info("Received empty PoseArray; nothing to plan.")
            self.pub_result.publish(Bool(data=True))
            return

        # Compute IK for each pose up front
        joint_trajectories = []
        for idx, pose in enumerate(msg.poses):
            try:
                joint_positions = self._ik_solver(pose)
            except Exception as e:
                self.get_logger().error(f"IK solver exception at waypoint {idx}: {e}")
                self.pub_result.publish(Bool(data=False))
                return

            if joint_positions is None or len(joint_positions) != len(self.joint_names):
                self.get_logger().error(f"IK solver returned an invalid solution for waypoint {idx}")
                self.pub_result.publish(Bool(data=False))
                return

            joint_trajectories.append(joint_positions)

        # Construct a single FollowJointTrajectory.Goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        # Stamp the header with current time (controller should start ASAP)
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        # Build JointTrajectoryPoint list with increasing time_from_start
        dt_between = 1.0  # seconds between consecutive waypoints
        points = []
        for i, q in enumerate(joint_trajectories):
            pt = JointTrajectoryPoint()
            pt.positions = q
            # Each point occurs (i+1) * dt_between seconds after start
            pt.time_from_start = Duration(sec=int((i + 1) * dt_between), nanosec=0)
            points.append(pt)

        goal_msg.trajectory.points = points

        # Send the goal asynchronously
        self.get_logger().info("Sending multi-point trajectory to controller...")
        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)


# created function for getting goal response callback 
    def _goal_response_callback(self, future):
        """
        Called when the action server accepts or rejects the trajectory goal.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal was rejected by the server.")
            self.pub_result.publish(Bool(data=False))
            return

        self.get_logger().info("Trajectory goal accepted; waiting for result...")
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._get_result_callback)


# created function for getting feeback call back 
    def _feedback_callback(self, feedback_msg):
        """
        Called whenever the action server publishes feedback during trajectory execution.
        """
        self.get_logger().info(f"Received feedback: {feedback_msg.feedback}")


# created function for getting result callback
    def _get_result_callback(self, future):
        """
        Called once the trajectory execution is complete (success or failure).
        Publish True if error_code == 0, otherwise False.
        """
        result = future.result().result
        if result.error_code == 0:
            self.get_logger().info("Trajectory executed successfully.")
            self.pub_result.publish(Bool(data=True))
        else:
            err_str = result.error_string if hasattr(result, 'error_string') else ""
            self.get_logger().error(
                f"Trajectory execution failed: error_code={result.error_code}, error_string='{err_str}'"
            )
            self.pub_result.publish(Bool(data=False))

# main function
def main(args=None):
    rclpy.init(args=args)
    node = PlanMotionLinearViaPoints()

    # Keep the node alive to process incoming PoseArray messages
    rclpy.spin(node)
    rclpy.shutdown()

# calling main function
if __name__ == '__main__':
    main()
