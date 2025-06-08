import os # imported os module
import numpy as np # imported numpy module

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.action import ActionClient # imported ActionClient

from control_msgs.action import FollowJointTrajectory # imported FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState # imported Joinstate
from builtin_interfaces.msg import Duration # imported Duration

from ikpy.chain import Chain # imported ik


# created class for URDF handler
class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.chain = None

# created function for loading chain
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


# created function for triming the chain end effector

    def trim_chain_to_end_effector(self, end_effector_name: str):
        if not self.chain:
            raise RuntimeError("Chain is not loaded. Call load_chain() first.")
        end_index = next((i for i, link in enumerate(self.chain.links)
                          if link.name == end_effector_name), None)
        if end_index is None:
            raise ValueError(f"End-effector link '{end_effector_name}' not found")
        # Keep only up to (and including) the EE link
        self.chain.links = self.chain.links[: end_index + 1]
        self.chain.active_links_mask = self.chain.active_links_mask[: end_index + 1]
        print(f"[URDFChainHandler] Chain trimmed to end-effector: {end_effector_name}")

# created function for inverse kinematics

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

        print("[URDFChainHandler] IK solution for target", target_position, ":")
        for i, angle in enumerate(joint_angles):
            print(f"  Joint {i} ({self.chain.links[i].name}): {angle:.4f} rad")
        return joint_angles

# created class for MovejointToclient node

class MoveJointToJointClient(Node):

    def __init__(self):
        super().__init__('move_joint_to_joint_client')

        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )
        self._current_joint_state = None

        # Subscribe to /joint_states so we know the joint names and latest positions
        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

# created function for state callback
    def _joint_state_callback(self, msg: JointState):

        # Always keep the latest JointState message
        self._current_joint_state = msg

# created function for sending the goal through points
    def send_via_points_goal(self,
                             via_joint_angles: list[np.ndarray],
                             total_duration: float):
        """
        Build a FollowJointTrajectory.Goal that passes through each
        via_joint_angles[i] at evenly spaced time_from_start stamps
        up to total_duration.

        via_joint_angles: a list of 1D numpy arrays (length = number of joints).
        total_duration: total time (in seconds) from the first to the last via point.
        """

        # Make sure we already have a current JointState (for joint ordering)
        if self._current_joint_state is None:
            self.get_logger().error("No joint_state received yet; cannot send trajectory.")
            return

        joint_names = self._current_joint_state.name[:]  # e.g. ["joint1", "joint2", ...]
        num_joints = len(joint_names)
        num_points = len(via_joint_angles)
        if num_points < 2:
            self.get_logger().error("Need at least 2 via points to plan a trajectory.")
            return

        # Check each via_angles array has same length as joint_names
        for idx, via in enumerate(via_joint_angles):
            if len(via) < num_joints:
                raise ValueError(f"Via point #{idx} has length {len(via)}, "
                                 f"but expected at least {num_joints}.")

        # Create goal message:
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        # Evenly space time stamps from t=0 to t=total_duration
        # e.g. if 3 via points, times = [0.0, total_duration/2, total_duration]
        times = np.linspace(0.0, total_duration, num_points)

        trajectory_points: list[JointTrajectoryPoint] = []
        for i, (config, t) in enumerate(zip(via_joint_angles, times)):
            pt = JointTrajectoryPoint()
            # Position: take only the first len(joint_names) entries from the IK output
            pt.positions = list(config[:num_joints])
            # Here we could also fill in velocities/accelerations if desired
            sec = int(t)
            nsec = int((t - sec) * 1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            trajectory_points.append(pt)

            self.get_logger().info(f"  → Via point {i}: time {t:.2f}s, "
                                   f"positions = {pt.positions}")

        goal_msg.trajectory.points = trajectory_points

        # Send the goal
        self.get_logger().info("Waiting for action server...")
        self._client.wait_for_server()

        self.get_logger().info(f"Sending trajectory with {num_points} points, "
                               f"total_duration={total_duration:.2f}s...")
        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

# cretead function for sending goal for callback
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Trajectory goal was rejected by the server.')
            return

        self.get_logger().info('Goal accepted → waiting for result...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

# created function for feeedback callback
    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

# created function for getting the result callback
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: error_code = {result.error_code}')
        # After getting the result, we can shut down
        rclpy.shutdown()


# crearted the main function

def main(args=None):

    rclpy.init(args=args)

    #  Set up IK handler
    urdf_path = "/home/midhun.eldose/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf"
    handler = URDFChainHandler(urdf_path)
    handler.load_chain()

    cartesian_targets = [
        np.array([0.3,  0.0, 0.2]),
        np.array([0.3,  0.2, 0.2]),
        np.array([0.1, -0.1, 0.3]),
    ]

    #  For each Cartesian target, compute an IK solution in joint space
    via_joint_configs: list[np.ndarray] = []
    prev_joints = None
    for idx, target in enumerate(cartesian_targets):
        # Use previous joints as initial guess (for continuity)
        ik_sol = handler.inverse_kinematics(target_position=target,
                                            initial_joints=prev_joints)
        via_joint_configs.append(ik_sol)
        prev_joints = ik_sol.copy()

    #  Create the action client
    move_joint = MoveJointToJointClient()

    # Wait until we have at least one JointState message
    while rclpy.ok() and move_joint._current_joint_state is None:
        rclpy.spin_once(move_joint,timeout_sec=0.1)


    total_duration = 12.0  # e.g. spend 12 seconds traversing all via points
    move_joint.send_via_points_goal(via_joint_angles=via_joint_configs,
                                    total_duration=total_duration)

    # Spin until the action completes (or node shuts down)
    rclpy.spin(move_joint)

# calling up the mian function
if __name__ == "__main__":
    main()
