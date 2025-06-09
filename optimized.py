#!/usr/bin/env python3
import os
import time
from typing import List, Optional, Tuple, Union, Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from builtin_interfaces.msg import Duration

from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R

# Optional CORBA for clear_ids()
try:
    from omniORB import CORBA
    from MairaCorba import Component
    _CORBA_AVAILABLE = True
except ImportError:
    _CORBA_AVAILABLE = False


def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angles into [-π, π)."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi


class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.chain = Chain.from_urdf_file(urdf_path, base_elements=[base_link])
        # by default, actuated joints are links 2..8
        self.actuated_indices = list(range(2, 9))

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        initial_joints: np.ndarray = None
    ) -> np.ndarray:
        if initial_joints is None:
            initial_joints = np.zeros(len(self.chain.links))
        return self.chain.inverse_kinematics(
            target_position,
            initial_position=initial_joints
        )


class IDManager:
    """Simple incrementing ID generator."""
    def __init__(self):
        self._id = 0

    def update_id(self) -> int:
        self._id += 1
        return self._id


class PlannerProgram:
    """Stub planner interface."""
    class _Cmd:
        Linear = object()
    def __init__(self):
        self.cmd = self._Cmd()
        self._last_joint = {}

    def set_command(self, cmd, cmd_id: int, current_joint_angles: List[float], reusable_id: int, **kwargs):
        # store last solution for stub
        self._last_joint[cmd_id] = current_joint_angles

    def get_last_joint_configuration(self, plan_id: int) -> List[float]:
        return self._last_joint.get(plan_id, [])

    def get_plan_status(self, plan_id: int) -> int:
        # always succeed stub
        return 1  # 1 = Success, 0 = Failed


class MairaKinematics(Node):
    def __init__(
        self,
        urdf_path: str,
        program: PlannerProgram,
        id_manager: IDManager,
        robot_interface: Any = None,
        base_link: str = "maira7M_root_link"
    ):
        super().__init__('maira_kinematics')

        # IK
        self.urdf_handler = URDFChainHandler(urdf_path, base_link)

        # Trajectory action client
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        # Joint state subscriber
        self._current_state: Optional[JointState] = None
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Planning stubs
        self._program = program
        self._id_manager = id_manager
        self._robot = robot_interface

        # Defaults
        self.speed_move_linear = 0.1
        self.acc_move_linear = 0.05
        self.num_waypoints = 10
        self._last_error_code: Optional[int] = None
        self._waiting_for_result = False


    # ------------------------
    # Core Callbacks
    # ------------------------
    def joint_state_callback(self, msg: JointState):
        """Cache the first received joint state."""
        if self._current_state is None:
            self._current_state = msg

    def feedback_callback(self, feedback_msg):
        """Generic feedback: log desired vs actual."""
        fb = feedback_msg.feedback
        for name, d, a in zip(fb.joint_names, fb.desired.positions, fb.actual.positions):
            self.get_logger().info(f"[feedback] {name}: desired={d:.4f}, actual={a:.4f}")

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by server.")
            self._waiting_for_result = False
            return
        self.get_logger().info("Goal accepted, awaiting result...")
        goal_handle.get_result_async().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self._last_error_code = result.error_code
        self.get_logger().info(f"Result received: error_code={result.error_code}")
        self._waiting_for_result = False


    # ------------------------
    # State Queries
    # ------------------------
    def get_current_joint_state(self) -> List[float]:
        """Return current joint positions."""
        if self._current_state is None:
            raise RuntimeError("No joint state received yet")
        return list(self._current_state.position)

    def get_current_cartesian_pose(self) -> List[float]:
        """Forward-kinematics from last joint state → [x,y,z, roll,pitch,yaw]."""
        if self._current_state is None:
            raise RuntimeError("No joint state received yet")
        # build full link vector
        full = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            full[idx] = self._current_state.position[i]
        T = self.urdf_handler.chain.forward_kinematics(full)
        xyz = T[:3, 3].tolist()
        rpy = R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False).tolist()
        return xyz + rpy


    # ------------------------
    # Simple Move Helpers
    # ------------------------
    def send_joint_trajectory(self, joint_names: List[str], joint_positions: List[float], duration: float):
        """Single-point FollowJointTrajectory goal."""
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = joint_names

        pt = JointTrajectoryPoint()
        pt.positions = joint_positions
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        traj.points = [pt]

        goal.trajectory = traj
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available.")
            return False

        self._waiting_for_result = True
        send_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)
        # wait until result callback flips flag
        while rclpy.ok() and self._waiting_for_result:
            rclpy.spin_once(self, timeout_sec=0.1)
        return (self._last_error_code == FollowJointTrajectory.Result().SUCCESSFUL)


    # ------------------------
    # Cartesian ↔ Joint Moves
    # ------------------------
    def cartesian_to_joint(self, goal_state: JointState, duration: float):
        """Helper to wrap existing JointState into a trajectory."""
        return self.send_joint_trajectory(goal_state.name, list(goal_state.position), duration)

    def move_joint_to_cartesian(self, goal_pose: List[float], duration: float = 5.0) -> bool:
        """Compute IK for [x,y,z] and send joint goal."""
        if self._current_state is None:
            self.get_logger().error("No joint state yet")
            return False
        # seed full
        full = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            full[idx] = self._current_state.position[i]
        sol = self.urdf_handler.inverse_kinematics(np.array(goal_pose), full)
        actuated = normalize_to_pi(sol[self.urdf_handler.actuated_indices])
        names = [f"joint{i+1}" for i in range(len(actuated))]
        return self.send_joint_trajectory(names, actuated.tolist(), duration)

    def get_elbow_up_ik_solution(self, target_pos: np.ndarray) -> np.ndarray:
        """Try elbow‐up seeds, fallback to first valid."""
        if self._current_state is None:
            raise RuntimeError("No joint state yet")
        base = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            base[idx] = self._current_state.position[i]
        seeds = [(0.0,0.0), (0.0,np.pi), (np.pi,0.0), (np.pi,np.pi)]
        first = None
        for d6, d7 in seeds:
            seed = base.copy()
            i6, i7 = self.urdf_handler.actuated_indices[4], self.urdf_handler.actuated_indices[5]
            seed[i6] += ((seed[i6]<0)*2-1)*d6
            seed[i7] += ((seed[i7]<0)*2-1)*d7
            try:
                sol = self.urdf_handler.inverse_kinematics(target_pos, seed)
                return sol
            except Exception:
                if first is None:
                    try:
                        first = self.urdf_handler.inverse_kinematics(target_pos, seed)
                    except Exception:
                        pass
        if first is not None:
            return first
        raise ValueError("No IK solution found")

    def ik_solver(self, goal_pose: List[float]) -> Optional[List[float]]:
        """Wrap IK solver and return actuated-joint list."""
        sol = self.get_elbow_up_ik_solution(np.array(goal_pose[:3]))
        actuated = sol[self.urdf_handler.actuated_indices]
        return normalize_to_pi(np.array(actuated)).tolist()


    # ------------------------
    # Joint ↔ Joint Moves
    # ------------------------
    def move_joint_to_joint(self, target_positions: List[float], duration: float = 1.0) -> bool:
        """Simple one-point joint → joint move."""
        if self._current_state is None:
            self.get_logger().warn("No joint state yet")
            return False
        names = list(self._current_state.name)
        return self.send_joint_trajectory(names, target_positions, duration)

    def is_trajectory_valid(self, trajectory: List[List[float]]) -> bool:
        """Check each waypoint length matches current joint count."""
        if self._current_state is None:
            self.get_logger().warn("No joint state yet")
            return False
        n = len(self._current_state.name)
        for wp in trajectory:
            if not isinstance(wp, list) or len(wp) != n:
                self.get_logger().error(f"Invalid waypoint length: expected {n}, got {len(wp)}")
                return False
        return True


    # ------------------------
    # Plan‐Motion Stubs
    # ------------------------
    def plan_motion_linear(
        self,
        goal_pose: List[float],
        start_cartesian_pose: List[float] = None,
        start_joint_states: List[float] = None,
        speed: float = None,
        acc: float = None,
        reusable: bool = False
    ) -> Tuple[Tuple[bool,bool], int, List[float]]:
        """Stub: delegate to PlannerProgram."""
        if start_cartesian_pose is None or start_joint_states is None:
            start_cartesian_pose = self.get_current_cartesian_pose()
            start_joint_states = self.get_current_joint_state()
        pid = self._id_manager.update_id()
        self._program.set_command(
            self._program.cmd.Linear,
            cmd_id=pid,
            current_joint_angles=start_joint_states,
            reusable_id=int(reusable),
            target_pose=[start_cartesian_pose, goal_pose],
            speed=speed or self.speed_move_linear,
            acceleration=acc or self.acc_move_linear,
            blending=False,
            blend_radius=0.0
        )
        flags = self.is_id_successful(pid)
        last = self._program.get_last_joint_configuration(pid) if all(flags) else []
        return flags, pid, last

    def plan_motion_joint_to_joint(
        self,
        goal_pose: List[float],
        start_joint_states: List[float] = None,
        speed: float = None,
        acc: float = None,
        reusable: bool = False
    ) -> Tuple[Tuple[bool,bool], int, List[float]]:
        """Stub: joint‐to‐joint planning."""
        if start_joint_states is None:
            start_joint_states = self.get_current_joint_state()
        pid = self._id_manager.update_id()
        self._program.set_command(
            self._program.cmd.Linear,
            cmd_id=pid,
            current_joint_angles=start_joint_states,
            reusable_id=int(reusable),
            target_pose=[start_joint_states, goal_pose],
            speed=speed or self.speed_move_linear,
            acceleration=acc or self.acc_move_linear,
            blending=False,
            blend_radius=0.0
        )
        flags = self.is_id_successful(pid)
        last = self._program.get_last_joint_configuration(pid) if all(flags) else []
        return flags, pid, last

    def plan_motion_linear_via_points(
        self,
        goal_poses: List[List[float]],
        start_cartesian_pose: List[float] = None,
        start_joint_states: List[float] = None,
        speed: float = None,
        acc: float = None,
        blending_radius: float = None,
        reusable: bool = False
    ) -> Tuple[Tuple[bool,bool], int, List[float]]:
        """Stub: linear via‐points planning."""
        if start_cartesian_pose is None or start_joint_states is None:
            start_cartesian_pose = self.get_current_cartesian_pose()
            start_joint_states = self.get_current_joint_state()
        all_targets = [start_cartesian_pose] + goal_poses
        pid = self._id_manager.update_id()
        self._program.set_command(
            self._program.cmd.Linear,
            cmd_id=pid,
            current_joint_angles=start_joint_states,
            reusable_id=int(reusable),
            target_pose=all_targets,
            speed=speed or self.speed_move_linear,
            acceleration=acc or self.acc_move_linear,
            blending=True,
            blend_radius=blending_radius or 0.01
        )
        flags = self.is_id_successful(pid)
        last = self._program.get_last_joint_configuration(pid) if all(flags) else []
        return flags, pid, last


    # ------------------------
    # “Real” Trajectory via Points
    # ------------------------
    def move_linear_via_points(
        self,
        positions_list: List[List[float]],
        times_list: List[float],
        speed_scale: float = 1.0
    ) -> bool:
        """Send a multi‐point Cartesian linear move (via IK)."""
        # first, compute IK sequence
        if len(positions_list) != len(times_list):
            raise ValueError("positions and times length mismatch")
        joint_traj: List[List[float]] = []
        for pose in positions_list:
            js = self.ik_solver(pose)
            if js is None:
                raise RuntimeError(f"IK failed for {pose}")
            joint_traj.append(js)
        # then send as joint‐via‐points
        return self.move_joint_via_points(joint_traj, times_list, speed_scale)

    def move_joint_via_points(
        self,
        positions_list: List[List[float]],
        times_list: List[float],
        speed_scale: float = 1.0
    ) -> bool:
        """Send a multi‐point joint trajectory."""
        if not self.is_trajectory_valid(positions_list):
            return False
        traj = JointTrajectory()
        traj.joint_names = list(self._current_state.name)
        prev_pos = None
        prev_time = 0.0
        for t_raw, pos in zip(times_list, positions_list):
            t = t_raw / speed_scale
            pt = JointTrajectoryPoint()
            pt.positions = pos
            dt = 1e-6 if prev_pos is None else max(t - prev_time, 1e-6)
            pt.velocities = [ (p - pp)/dt for p, pp in zip(pos, prev_pos or pos) ]
            sec = int(t); nsec = int((t-sec)*1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)
            traj.points.append(pt)
            prev_pos = pos; prev_time = t
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        self._waiting_for_result = True
        send_fut = self._client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        send_fut.add_done_callback(self.goal_response_callback)
        while rclpy.ok() and self._waiting_for_result:
            rclpy.spin_once(self, timeout_sec=0.1)
        return (self._last_error_code == FollowJointTrajectory.Result().SUCCESSFUL)


    # ------------------------
    # ID / CORBA Helpers
    # ------------------------
    def is_id_successful(self, plan_id: int, timeout: float = 10.0) -> Tuple[bool,bool]:
        """Poll PlannerProgram.get_plan_status until Success/Failed."""
        start = time.time()
        status = self._program.get_plan_status(plan_id)
        while status not in (0, 1) and (time.time() - start) < timeout:
            time.sleep(0.01)
            status = self._program.get_plan_status(plan_id)
        return (status == 1, status == 1)

    def is_plan_successful(self, plan_id: int, timeout: float = 10.0) -> bool:
        ok, _ = self.is_id_successful(plan_id, timeout)
        return ok

    def clear_ids(self, ids: List[int]) -> bool:
        """Clear given plan IDs from robot via CORBA."""
        if not _CORBA_AVAILABLE or self._robot is None:
            self.get_logger().warn("CORBA not available or robot interface missing")
            return False
        try:
            rts = Component(self._robot, "RTS")
            seq = CORBA.Any(CORBA.TypeCode("IDL:omg.org/CORBA/DoubleSeq:1.0"), ids)
            return (rts.callService("clearSplineId", [seq]) == 0)
        except Exception as e:
            self.get_logger().error(f"Error clearing IDs: {e}")
            return False


def main(args=None):
    rclpy.init(args=args)
    urdf = os.path.expanduser(
        '~/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf'
    )
    program = PlannerProgram()
    idm = IDManager()
    node = MairaKinematics(urdf, program, idm, robot_interface=None)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
