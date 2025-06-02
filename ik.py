
import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration

from ikpy.chain import Chain

class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.chain = None

    def load_chain(self):
        if not os.path.isfile(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        self.chain = Chain.from_urdf_file(
            self.urdf_path,
            base_elements=[self.base_link]
        )
        print("[URDFChainHandler] Loaded chain links:")
        for i, link in enumerate(self.chain.links):
            print(f"  {i}: {link.name}")

    def trim_chain_to_end_effector(self, end_effector_name: str):
        if not self.chain:
            raise RuntimeError("Chain is not loaded. Call load_chain() first.")
        end_index = next((i for i, link in enumerate(self.chain.links)
                          if link.name == end_effector_name), None)
        if end_index is None:
            raise ValueError(f"End-effector link '{end_effector_name}' not found")
        self.chain.links = self.chain.links[: end_index + 1]
        self.chain.active_links_mask = self.chain.active_links_mask[: end_index + 1]
        print(f"[URDFChainHandler] Chain trimmed to end-effector: {end_effector_name}")

    def inverse_kinematics(self, target_position: np.ndarray,
                           initial_joints: np.ndarray = None) -> np.ndarray:
        if not self.chain:
            raise RuntimeError("Chain is not loaded. Call load_chain() first.")
        if initial_joints is None:
            initial_joints = np.zeros(len(self.chain.links))
        joint_angles = self.chain.inverse_kinematics(
            target_position,
            initial_position=initial_joints
        )
        print("[URDFChainHandler] IK solution:")
        for i, angle in enumerate(joint_angles):
            print(f"  Joint {i} ({self.chain.links[i].name}): {angle:.4f} rad")
        return joint_angles


class MoveJointToJointClient(Node):
    def __init__(self):
        super().__init__('move_joint_to_joint_client')

        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )
        self._current_joint_state = None

        # Subscribe to /joint_states so we know the joint names and positions
        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

    def _joint_state_callback(self, msg: JointState):
        # Keep overwriting so that _current_joint_state always has the latest
        self._current_joint_state = msg

    def send_goal(self, goal_joint_state: JointState, duration: float):
        # Build a FollowJointTrajectory.Goal()
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = goal_joint_state.name[:]  # copy the names
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        point = JointTrajectoryPoint()
        point.positions = list(goal_joint_state.position[:])

        # Convert the float `duration` to sec/nsec
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal_msg.trajectory.points = [point]

        self.get_logger().info('Waiting for action server...')
        self._client.wait_for_server()

        self.get_logger().info(f'Sending goal (duration={duration:.2f}s)...')
        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by the server.')
            return

        self.get_logger().info('Goal accepted → waiting for result...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: error_code = {result.error_code}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    urdf_path = "/home/midhun.eldose/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf"
    handler = URDFChainHandler(urdf_path)
    handler.load_chain()
    # handler.trim_chain_to_end_effector("LeftWristY")  # if needed

    # Compute IK solution (as a NumPy array)
    target_pos = np.array([-5.0, 0.6, 0.5])
    joint_angles = handler.inverse_kinematics(target_pos)

    move_joint = MoveJointToJointClient()

    # Wait until we actually receive at least one /joint_states message
    # so that move_joint._current_joint_state is not None:
    while rclpy.ok() and move_joint._current_joint_state is None:
        rclpy.spin_once(move_joint, timeout_sec=0.1)

    # Now build a new JointState message that reuses the joint names,
    # but replaces the positions with our IK solution.
    js = JointState()
    js.name = move_joint._current_joint_state.name[:]       # copy the names
    js.position = list(joint_angles[:len(js.name)])         # truncate / pad as needed

    # Pick some reasonable duration (for example, 2.0 seconds)
    duration = 2.0

    # Now actually send the goal
    move_joint.send_goal(js, duration)

    rclpy.spin(move_joint)


if __name__ == "__main__":
    main()
