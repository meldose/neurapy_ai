#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from your_package.action import MoveJointToJoint  # Replace with your actual package

from typing import List, Tuple

# Neura database service
from neura_ai_database_msgs.srv import ReadTCPPose, ReadTCPPoseRequest

# Neura utilities
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils.ros_conversions import geometry_msg_pose_2_pose


class DatabaseClientROS2:
    """ROS 2 version of the Neura AI database client for retrieving joint positions."""

    def __init__(self, node: Node):
        self.node = node
        self.cli_tcp_pose = self.node.create_client(ReadTCPPose, '/neura_ai_database/read_tcpPose')

    def wait_for_services(self):
        self.node.get_logger().info(" Waiting for database service...")
        while not self.cli_tcp_pose.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn('/neura_ai_database/read_tcpPose not available, retrying...')

    def get_joint_positions(self, point_name: str) -> Tuple[ReturnCode, List[float]]:
        """Fetch joint positions stored in the Neura AI database under a point name."""
        req = ReadTCPPoseRequest()
        req.tcp_point_name = point_name

        future = self.cli_tcp_pose.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None:
            res = future.result()
            if res.return_code.value < 0:
                return ReturnCode(res.return_code.value, res.return_code.message), []
            return ReturnCode(), list(res.tcp_pose.tcp_pose_joint_space)
        else:
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, "Service call failed"), []


class MoveJointToJointClient(Node):
    """ROS 2 Action Client to move the robot to joint state positions from the database."""

    def __init__(self):
        super().__init__('move_joint_to_joint_client')
        self._action_client = ActionClient(self, MoveJointToJoint, 'move_joint_to_joint')
        self._db_client = DatabaseClientROS2(self)
        self._db_client.wait_for_services()

    def send_goal_from_database(self, point_name: str):
        """Send a goal to the action server using joint values from the Neura database."""
        retcode, joints = self._db_client.get_joint_positions(point_name)
        if not retcode:
            self.get_logger().error(f" Failed to retrieve joint data: {retcode.message}")
            return

        joint_state = JointState()
        joint_state.name = [f'joint_{i+1}' for i in range(len(joints))]
        joint_state.position = joints

        goal_msg = MoveJointToJoint.Goal()
        goal_msg.target_joint_state = joint_state

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info(' Goal rejected by server.')
            return
        self.get_logger().info(' Goal accepted.')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f' Feedback: {feedback_msg.feedback.status}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f' Move complete: success={result.success}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointClient()

    # Replace with your actual point name stored in the Neura AI database
    point_name = "home_position"
    node.send_goal_from_database(point_name)

    rclpy.spin(node)


if __name__ == '__main__':
    main()
