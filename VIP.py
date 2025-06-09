
###################################################################
 CARTESIAN TO JOINT
####################################################################

import os # imported os module
import numpy as np # imported numpy module

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.action import ActionClient #imported Actionclient
from typing import List, Optional, Tuple, Union

from control_msgs.action import FollowJointTrajectory # imported FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint # imported JoinTrajectory
from sensor_msgs.msg import JointState #imported Joinstate
from builtin_interfaces.msg import Duration # imported Duration

from ikpy.chain import Chain # imported ik

# created class for URDF handler
class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.chain = None

# function for loading chain
    def load_chain(self):
        if not os.path.isfile(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        self.chain = Chain.from_urdf_file(
            self.urdf_path,
            base_elements=[self.base_link]
        )


        print("[URDFChainHandler] Hard coded joint indices:")
        for i in range(2, 9):
            print(f"  {i}: maira7M_joint{i-1}")


# created function for inverse kinematics

    def inverse_kinematics(self, target_position: np.ndarray,
                           initial_joints: np.ndarray = None) -> np.ndarray:
        if not self.chain:
            raise RuntimeError("Chain is not loaded. Call load_chain() first.")
        if initial_joints is None:
            initial_joints = np.zeros(len(self.chain.links))

        full_solution = self.chain.inverse_kinematics(
            target_position,
            initial_position=initial_joints
        )

        print("[URDFChainHandler] Raw IK solution (one value per link):")
        for i, angle in enumerate(full_solution):
            print(f"  Joint {i} ({self.chain.links[i].name}): {angle:.4f} rad")

        return full_solution

class CartesiantoJoint(Node):

    def __init__(self,
                 joint_names: List[str],
                 goal_angles: np.ndarray,
                 duration: float):
        super().__init__('move_joint_to_joint_client')

        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )
        self._current_joint_state = None
        self._sent_goal = False

        # Store exactly what you passed in
        self._joint_names = joint_names
        self._goal_poses: np.ndarray = goal_angles
        self._duration = duration

        # Optional params (no trailing commas!)
        self.acc: Optional[float] = None
        self.rot_acc: Optional[float] = None
        self.blending_radius: Optional[float] = None

        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def joint_state_callback(self, msg: JointState):
        if self._sent_goal:
            return

        # … logging omitted …

        # Build a JointState just for the 7 joints we want to move:
        js = JointState()
        js.name = self._joint_names[:]
        js.position = [float(a) for a in self._goal_poses]

        # ← call the correct method
        self.cartesian_to_joint(js, self._duration)
        self._sent_goal = True
        self.destroy_subscription(self._js_sub)

    def cartesian_to_joint(self, goal_joint_state: JointState, duration: float):
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = goal_joint_state.name[:]
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        point = JointTrajectoryPoint()
        point.positions = list(goal_joint_state.position[:])

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


# setting the function for giving goal response callback
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by the server.')
            return

        self.get_logger().info('Goal accepted → waiting for result...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

# creating function for feedback callback
    def feedback_callback(self, feedback_msg):
        names = feedback_msg.feedback.joint_names
        desired = feedback_msg.feedback.desired.positions
        actual = feedback_msg.feedback.actual.positions
        for i, nm in enumerate(names):
            self.get_logger().info(
                f"[feedback] '{nm}': desired={desired[i]:.4f}, actual={actual[i]:.4f}"
            )

# creating function for getting result callback
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: error_code = {result.error_code}')
        rclpy.shutdown()


def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi

#  creating main function :

def main(args=None):
    rclpy.init(args=args)

    urdf_path = "/home/midhun.eldose/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf"
    handler = URDFChainHandler(urdf_path)
    handler.load_chain()

    goal_pose_cartesian = np.array([-0.5, -0.4, 1.0])
    full_joint_angles = handler.inverse_kinematics(goal_pose_cartesian)

    #  Extract only the 7 actuated revolute joints (indices 2..8)
    raw_actuated = full_joint_angles[2 : 2 + 7]

    #  Normalize each to [−π, +π)
    wrapped_actuated = normalize_to_pi(raw_actuated)

    #  Hard‐code the 7 actuated joint names in the exact order:
    actuated_names = [
        "joint1",  # index 2
        "joint2",  # index 3
        "joint3",  # index 4
        "joint4",  # index 5
        "joint5",  # index 6
        "joint6",  # index 7
        "joint7",  # index 8
    ]

    #  Instantiate the MoveJoint client with those 7 hard‐coded names + angles
    duration = 5.0
    move_joint = CartesiantoJoint(actuated_names, wrapped_actuated, duration)

    #  Spin until we send the goal:
    rclpy.spin(move_joint)


if __name__ == "__main__":
    main()

###################################################################
CLEAR IDS
####################################################################

import os #imported os module
import time #imported time module
from typing import List, Tuple, Union, Optional, Any

import numpy as np # imported numpy module
import rclpy #imported rclpy
from rclpy.node import Node #imported Node
from rclpy.action import ActionClient  # imported Actionclient
from sensor_msgs.msg import JointState # imported jointstate
from trajectory_msgs.msg import JointTrajectoryPoint # imported JOintTrajectory
from control_msgs.action import FollowJointTrajectory # imported followjointTrajectory
from builtin_interfaces.msg import Duration # imported Duration
from ikpy.chain import Chain # imported ik
from scipy.spatial.transform import Rotation as R # imported Rotation

try:
    from omniORB import CORBA, PortableServer
    import CosNaming
    from MairaCorba import Component
    _CORBA_AVAILABLE = True
except ImportError:
    _CORBA_AVAILABLE = False


class calculation:
    Success = 1
    Failed = 0

class IDManager:
    def __init__(self):
        self._id = 0
    def update_id(self) -> int:
        self._id += 1
        return self._id

class PlannerProgram:
    class _Cmd:
        Linear = object()
    def __init__(self):
        self.cmd = self._Cmd()
        self._last_joint = {}
    def set_command(self,
                    cmd,
                    cmd_id: int,
                    current_joint_angles: List[float],
                    reusable_id: int,
                    **kwargs):
        self._last_joint[cmd_id] = current_joint_angles
    def get_last_joint_configuration(self, plan_id: int) -> List[float]:
        return self._last_joint.get(plan_id, [])
    def get_plan_status(self, plan_id: int):
        return calculation.Success

# Helper: normalize angles into [-pi, pi)
def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi

# class for URDF chain handler
class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.chain = Chain.from_urdf_file(urdf_path, base_elements=[base_link])
        self.actuated_indices = list(range(2, 9))

# function for inverse kinematics
    def inverse_kinematics(self,
                           target_position: np.ndarray,
                           initial_joints: np.ndarray = None) -> np.ndarray:
        if initial_joints is None:
            initial_joints = np.zeros(len(self.chain.links))
        return self.chain.inverse_kinematics(
            target_position,
            initial_position=initial_joints
        )


# class MairaKinematics
class MairaKinematics(Node):
    def __init__(
        self,
        urdf_path: str,
        program: PlannerProgram,
        id_manager: IDManager,
        robot_interface: Any,
        base_link: str = "maira7M_root_link"
    ):
        super().__init__('maira_kinematics')
        # IK handler
        self.urdf_handler = URDFChainHandler(urdf_path, base_link)
        # Trajectory action client
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )
        # current joint state
        self._current_state: JointState = None
        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state, 10
        )
        # planner and robot
        self._program    = program
        self._id_manager = id_manager
        self._robot      = robot_interface
        # defaults
        self.speed_move_linear = 0.1
        self.acc_move_linear   = 0.05


# function for join state
    def joint_state(self, msg: JointState):
        self._current_state = msg
        self.destroy_subscription(self._js_sub)

# function for getting elbow ik solution
    def get_elbow_up_ik_solution(self, target_pos: np.ndarray) -> np.ndarray:
        if self._current_state is None:
            raise RuntimeError("No joint state yet")
        zeros = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            zeros[idx] = self._current_state.position[i]
        retry = [(0.0,0.0),(0.0,np.pi),(np.pi,0.0),(np.pi,np.pi)]
        first_valid = None
        for d6, d7 in retry:
            seed = zeros.copy()
            seed[self.urdf_handler.actuated_indices[4]] += ((seed[self.urdf_handler.actuated_indices[4]]<0)*2-1)*d6
            seed[self.urdf_handler.actuated_indices[5]] += ((seed[self.urdf_handler.actuated_indices[5]]<0)*2-1)*d7
            try:
                sol = self.urdf_handler.inverse_kinematics(target_pos, seed)
            except Exception as e:
                self.get_logger().debug(f"IK fail seed {seed}: {e}")
                continue
            if first_valid is None:
                first_valid = sol
        if first_valid is not None:
            self.get_logger().warn("Falling back to first IK sol")
            return first_valid
        raise ValueError("No IK solution found")

 # function for sending joint trajectory
    def send_joint_trajectory(self,
                              joint_names: List[str],
                              joint_positions: List[float],
                              duration: float):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = joint_positions
        sec  = int(duration)
        nsec = int((duration-sec)*1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [pt]
        self._client.wait_for_server()
        fut = self._client.send_goal_async(goal, feedback_callback=self.feedback)
        fut.add_done_callback(self.goal_response)

# function for feedback
    def feedback(self, feedback):
        self.get_logger().debug(f"Feedback: {feedback}")

# function for goal response
    def goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn("Goal rejected")
            return
        gh.get_result_async().add_done_callback(self.result)

# function for getting result
    def result(self, future):
        res = future.result().result
        self.get_logger().info(f"Result code: {res.error_code}")
        rclpy.shutdown()

# function for checking if id is successful or not
    def is_id_successful(self, plan_id: int, timeout: float = 10.0) -> Tuple[bool, bool]:
        try:
            start = time.time()
            status = self._program.get_plan_status(plan_id)
            while status not in (calculation.Failed, calculation.Success) \
                  and (time.time() - start) < timeout:
                time.sleep(0.01)
                status = self._program.get_plan_status(plan_id)
            return (status == calculation.Success, status == calculation.Success)
        except Exception as ex:
            self.get_logger().error(f"Motion Id {plan_id} doesn't exist: {ex}")
            raise

# function for checking if plan is successful or not
    def is_plan_successful(self, plan_id: int, timeout: float = 10.0) -> bool:
        success, _ = self.is_id_successful(plan_id, timeout)
        return success

# function for clearing ids
    def clear_ids(self, ids: List[int]) -> bool:
        """Clear given plan IDs from memory stack to prevent overload."""
        try:
            rts = Component(self._robot, "RTS")
            seq = CORBA.Any(CORBA.TypeCode("IDL:omg.org/CORBA/DoubleSeq:1.0"), ids)
            ok = (rts.callService("clearSplineId", [seq]) == 0)
            if not ok:
                self.get_logger().warning(f"Failed to clear IDs: {ids}")
            return ok
        except Exception as e:
            self.get_logger().error(f"Error clearing IDs {ids}: {e}")
            return False

# function for plan motion linear
    def plan_motion_linear(
        self,
        goal_pose: List[float],
        start_cartesian_pose: Union[List[float], None] = None,
        start_joint_states: Union[List[float], None]    = None,
        speed: Optional[float] = None,
        acc:   Optional[float] = None,
        reusable: Optional[bool] = False,
    ) -> Tuple[Tuple[bool, bool], int, List[float]]:
        if not all([start_cartesian_pose, start_joint_states]):
            start_cartesian_pose = self.get_current_cartesian_pose()
            start_joint_states   = self.get_current_state_positions()
        self.throw_if_pose_invalid(goal_pose)
        self.throw_if_pose_invalid(start_cartesian_pose)
        self.throw_if_joint_invalid(start_joint_states)
        props = {
            "target_pose": [start_cartesian_pose, goal_pose],
            "speed":        self.speed_move_linear if speed is None else speed,
            "acceleration": self.acc_move_linear   if acc   is None else acc,
            "blending":     False,
            "blend_radius": 0.0,
        }
        pid = self._id_manager.update_id()
        cmd = self._program.cmd.Linear
        self._program.set_command(
            cmd, cmd_id=pid,
            current_joint_angles=start_joint_states,
            reusable_id=1 if reusable else 0,
            **props
        )
        flags = self.is_id_successful(pid)
        last = None
        if all(flags):
            last = self._program.get_last_joint_configuration(pid)
        return flags, pid, last

# function for getting current state positions
    def get_current_state_positions(self) -> List[float]:
        return list(self._current_state.position)

# function for getting curretn cartesian pose
    def get_current_cartesian_pose(self) -> List[float]:
        if self._current_state is None:
            raise RuntimeError("No joint state yet")
        full = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            full[idx] = self._current_state.position[i]
        T = self.urdf_handler.chain.forward_kinematics(full)
        xyz = T[:3,3].tolist()
        rpy = R.from_matrix(T[:3,:3]).as_euler('xyz', degrees=False).tolist()
        return xyz + rpy

# function for checking if the pose is invalid or not
    def throw_if_pose_invalid(self, pose: List[float]):
        assert len(pose) == 6, "Pose must be list of 6 floats"

# checking if joint is invalid or not
    def throw_if_joint_invalid(self, joints: List[float]):
        exp = len(self.urdf_handler.actuated_indices)
        assert len(joints) == exp, f"Expected {exp} joint values"

# main function

def main(args=None):
    rclpy.init(args=args)
    program    = PlannerProgram()
    id_manager = IDManager()
    # Initialize your CORBA robot interface here
    robot_interface = None  # replace with actual robot handle

    urdf_path = os.path.expanduser(
        '~/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf'
    )
    node = MairaKinematics(urdf_path, program, id_manager, robot_interface)
    while rclpy.ok() and node._current_state is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    goal = [-0.5, -0.4, 1.0, 0.0, 0.0, 0.0]
    (ok, _), pid, joints = node.plan_motion_linear(goal)
    print(f"Plan {pid} success: {node.is_plan_successful(pid)}")


    if ok:
        cleared = node.clear_ids([pid])
        print(f"Cleared IDs [ {pid} ]: {cleared}")

    if ok and joints:
        names = [f"joint{i+1}" for i in range(len(joints))]
        traj  = normalize_to_pi(np.array(joints)).tolist()
        node.send_joint_trajectory(names, traj, duration=5.0)
        rclpy.spin(node)

# calling main function
if __name__ == '__main__':
    main()

####################################################################
#  IS ID SUCCESSFUL
#####################################################################

import os #imported os module
import time #imported time module
from typing import List, Tuple, Union, Optional, Any

import numpy as np # imported numpy module
import rclpy #imported rclpy
from rclpy.node import Node #imported Node
from rclpy.action import ActionClient  # imported Actionclient
from sensor_msgs.msg import JointState # imported jointstate
from trajectory_msgs.msg import JointTrajectoryPoint # imported JOintTrajectory
from control_msgs.action import FollowJointTrajectory # imported followjointTrajectory
from builtin_interfaces.msg import Duration # imported Duration
from ikpy.chain import Chain # imported ik
from scipy.spatial.transform import Rotation as R # imported Rotation

class calculation:

    Success = 1
    Failed = 0

class IDManager:
    """Stub ID manager: incrementing IDs."""
    def __init__(self):
        self._id = 0
    def update_id(self) -> int:
        self._id += 1
        return self._id

class PlannerProgram:
    """Stub planner: replace with your implementation."""
    class _Cmd:
        Linear = object()
    def __init__(self):
        self.cmd = self._Cmd()
        self._last_joint = {}
    def set_command(self,
                    cmd,
                    cmd_id: int,
                    current_joint_angles: List[float],
                    reusable_id: int,
                    **kwargs):
        self._last_joint[cmd_id] = current_joint_angles
    def get_last_joint_configuration(self, plan_id: int) -> List[float]:
        return self._last_joint.get(plan_id, [])
    def get_plan_status(self, plan_id: int):
        # stub: always immediately succeed
        return calculation.Success

# Helper: normalize angles into [-pi, pi)
def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi

# class for URDF chain handler
class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.chain = Chain.from_urdf_file(urdf_path, base_elements=[base_link])
        self.actuated_indices = list(range(2, 9))

# function for inverse kinematics
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

# class MairaKinematics
class MairaKinematics(Node):
    def __init__(
        self,
        urdf_path: str,
        program: PlannerProgram,
        id_manager: IDManager,
        base_link: str = "maira7M_root_link"
    ):
        super().__init__('maira_kinematics')
        self.urdf_handler = URDFChainHandler(urdf_path, base_link)
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )
        self._current_state: JointState = None
        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state, 10
        )
        self._program = program
        self._id_manager = id_manager
        self.speed_move_linear = 0.1
        self.acc_move_linear   = 0.05

# function for getting joint state
    def joint_state(self, msg: JointState):
        self._current_state = msg
        self.destroy_subscription(self._js_sub)

# function for getting elbow up ik solution

    def get_elbow_up_ik_solution(self, target_pos: np.ndarray) -> np.ndarray:
        if self._current_state is None:
            raise RuntimeError("No joint state yet")
        zeros = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            zeros[idx] = self._current_state.position[i]
        retry = [(0.0,0.0),(0.0,np.pi),(np.pi,0.0),(np.pi,np.pi)]
        first_valid = None
        for d6, d7 in retry:
            seed = zeros.copy()
            seed[self.urdf_handler.actuated_indices[4]] += ((seed[self.urdf_handler.actuated_indices[4]]<0)*2-1)*d6
            seed[self.urdf_handler.actuated_indices[5]] += ((seed[self.urdf_handler.actuated_indices[5]]<0)*2-1)*d7
            try:
                sol = self.urdf_handler.inverse_kinematics(target_pos, seed)
            except Exception as e:
                self.get_logger().debug(f"IK fail seed {seed}: {e}")
                continue
            if first_valid is None:
                first_valid = sol
        if first_valid is not None:
            self.get_logger().warn("Falling back to first IK sol")
            return first_valid
        raise ValueError("No IK solution found")

# function for sending joint trajectory
    def send_joint_trajectory(self, joint_names: List[str], joint_positions: List[float], duration: float):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        pt = JointTrajectoryPoint()
        pt.positions = joint_positions
        sec = int(duration); nsec = int((duration-sec)*1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [pt]
        self._client.wait_for_server()
        future = self._client.send_goal_async(goal, feedback_callback=self.feedback)
        future.add_done_callback(self.goal_response)

# function for feedback
    def feedback(self, feedback):
        self.get_logger().debug(f"Feedback: {feedback}")

# function for goal response
    def goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn("Goal rejected")
            return
        gh.get_result_async().add_done_callback(self.result)

#function for result
    def result(self, future):
        res = future.result().result
        self.get_logger().info(f"Result code: {res.error_code}")
        rclpy.shutdown()

# function for checking is id successful or not
    def is_id_successful(self, plan_id: int, timeout: float = 10.0) -> Tuple[bool, bool]:
        """Check if planning for given id succeeded or failed."""
        try:
            t_start = time.time()
            status = self._program.get_plan_status(plan_id)
            while (
                status != calculation.Failed
                and status != calculation.Success
                and (time.time() - t_start) < timeout
            ):
                time.sleep(0.01)
                status = self._program.get_plan_status(plan_id)
            return (status == calculation.Success, status == calculation.Success)
        except Exception as ex:
            self.get_logger().error(f"Motion Id {plan_id} doesn't exist: {str(ex)}")
            raise

# function for ehcking if plan is successful or not
    def is_plan_successful(self, plan_id: int, timeout: float = 10.0) -> bool:
        """Public helper to return True if the plan succeeded."""
        succeeded, _ = self.is_id_successful(plan_id, timeout)
        return succeeded


# function for plan motion linear
    def plan_motion_linear(
        self,
        goal_pose: List[float],
        start_cartesian_pose: Union[List[float], None] = None,
        start_joint_states: Union[List[float], None]    = None,
        speed: Optional[float] = None,
        acc:   Optional[float] = None,
        reusable: Optional[bool] = False,
    ) -> Tuple[Tuple[bool, bool], int, List[float]]:
        if not all([start_cartesian_pose, start_joint_states]):
            start_cartesian_pose = self.get_current_cartesian_pose()
            start_joint_states   = self.get_current_state_positions()
        self.throw_if_pose_invalid(goal_pose)
        self.throw_if_pose_invalid(start_cartesian_pose)
        self.throw_if_joint_invalid(start_joint_states)
        linear_prop = {
            "target_pose": [start_cartesian_pose, goal_pose],
            "speed":        self.speed_move_linear if speed is None else speed,
            "acceleration": self.acc_move_linear   if acc   is None else acc,
            "blending":     False,
            "blend_radius": 0.0,
        }
        plan_id = self._id_manager.update_id()
        cmd     = self._program.cmd.Linear
        self._program.set_command(
            cmd,
            **linear_prop,
            cmd_id=plan_id,
            current_joint_angles=start_joint_states,
            reusable_id=1 if reusable else 0
        )
        success_flags = self.is_id_successful(plan_id)
        last_state = None
        if all(success_flags):
            last_state = self._program.get_last_joint_configuration(plan_id)
        return success_flags, plan_id, last_state

# function for current state position
    def get_current_state_positions(self) -> List[float]:
        return list(self._current_state.position)

# function for curretn cartesain pose
    def get_current_cartesian_pose(self) -> List[float]:
        if self._current_state is None:
            raise RuntimeError("No joint state yet")
        full = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            full[idx] = self._current_state.position[i]
        T   = self.urdf_handler.chain.forward_kinematics(full)
        xyz = T[:3,3].tolist()
        rpy = R.from_matrix(T[:3,:3]).as_euler('xyz', degrees=False).tolist()
        return xyz + rpy

# function for pose is invalid or not
    def throw_if_pose_invalid(self, pose: List[float]):
        assert len(pose) == 6, "Pose must be list of 6 floats"

# function for checkijng if joint is invalid or not
    def throw_if_joint_invalid(self, joints: List[float]):
        exp = len(self.urdf_handler.actuated_indices)
        assert len(joints) == exp, f"Expected {exp} joint values"

# main function
def main(args=None):
    rclpy.init(args=args)
    program    = PlannerProgram()
    id_manager = IDManager()
    urdf_path = os.path.expanduser(
        '~/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf'
    )
    node = MairaKinematics(urdf_path, program, id_manager)
    while rclpy.ok() and node._current_state is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    goal_pose = [-0.5, -0.4, 1.0, 0.0, 0.0, 0.0]
    (plan_ok, exec_ok), pid, joints = node.plan_motion_linear(goal_pose)
    print(f"Plan {pid} succeeded? {plan_ok}")
    if node.is_plan_successful(pid):
        names = [f"joint{i+1}" for i in range(len(joints))]
        traj  = normalize_to_pi(np.array(joints)).tolist()
        node.send_joint_trajectory(names, traj, duration=5.0)
        rclpy.spin(node)

# calling main function
if __name__ == '__main__':
    main()

####################################################################
#  PLAN MOTION LINEAR
#####################################################################


import os #imported os module
import time #imported time module
from typing import List, Tuple, Union, Optional, Any

import numpy as np # imported numpy module
import rclpy #imported rclpy
from rclpy.node import Node #imported Node
from rclpy.action import ActionClient  # imported Actionclient
from sensor_msgs.msg import JointState # imported jointstate
from trajectory_msgs.msg import JointTrajectoryPoint # imported JOintTrajectory
from control_msgs.action import FollowJointTrajectory # imported followjointTrajectory
from builtin_interfaces.msg import Duration # imported Duration
from ikpy.chain import Chain # imported ik
from scipy.spatial.transform import Rotation as R # imported Rotation

# function to normalize
def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    """Normalize an array of angles to [-pi, pi)."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi

# class IDManager

class IDManager:
    """Stub ID manager. Replace with your own."""
    def __init__(self):
        self._id = 0
    def update_id(self) -> int:
        self._id += 1
        return self._id
    def is_id_successful(self, plan_id: int) -> Tuple[bool,bool]:
        # always successful in this stub
        return (True, True)

# class PLanner Porgram
class PlannerProgram:
    """Stub planner. Replace with your own."""
    class _Cmd:
        Linear = object()
    def __init__(self):
        self.cmd = self._Cmd()
        self._last_joint = {}
    def set_command(self,
                    cmd,
                    cmd_id: int,
                    current_joint_angles: List[float],
                    reusable_id: int,
                    **kwargs):
        # just store last joint config
        self._last_joint[cmd_id] = current_joint_angles
    def get_last_joint_configuration(self, plan_id: int) -> List[float]:
        return self._last_joint.get(plan_id, [])


# class URDF Chain Handler
class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.chain = Chain.from_urdf_file(urdf_path, base_elements=[base_link])
        self.actuated_indices = list(range(2, 9))  # joints 1–7

# function for inverse kinematics
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

#class Mairakinematics
class MairaKinematics(Node):
    def __init__(
        self,
        urdf_path: str,
        program: PlannerProgram,
        id_manager: IDManager,
        base_link: str = "maira7M_root_link"
    ):
        super().__init__('maira_kinematics')


        self.urdf_handler = URDFChainHandler(urdf_path, base_link)

        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )


        self._current_state: JointState = None
        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state, 10
        )


        self._program    = program
        self._id_manager = id_manager


        self.speed_move_linear = 0.1
        self.acc_move_linear   = 0.05

# function for joint states
    def joint_state(self, msg: JointState):
        self._current_state = msg
        self.destroy_subscription(self._js_sub)  # only need first

# functionf or elbow up ik solution
    def get_elbow_up_ik_solution(self, target_pos: np.ndarray) -> np.ndarray:
        if self._current_state is None:
            raise RuntimeError("No joint state yet")

        zeros = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            zeros[idx] = self._current_state.position[i]

        retry = [(0.0,0.0),(0.0,np.pi),(np.pi,0.0),(np.pi,np.pi)]
        first_valid = None
        for d6,d7 in retry:
            seed = zeros.copy()
            seed[self.urdf_handler.actuated_indices[4]] += ((seed[self.urdf_handler.actuated_indices[4]]<0)*2-1)*d6
            seed[self.urdf_handler.actuated_indices[5]] += ((seed[self.urdf_handler.actuated_indices[5]]<0)*2-1)*d7
            try:
                sol = self.urdf_handler.inverse_kinematics(target_pos, seed)
            except Exception:
                continue
            if first_valid is None:
                first_valid = sol

        if first_valid is not None:
            self.get_logger().warn("Falling back to first IK sol")
            return first_valid
        raise ValueError("No IK solution found")

# function for sending joint trajectory
    def send_joint_trajectory(self, joint_names: List[str], joint_positions: List[float], duration: float):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names

        pt = JointTrajectoryPoint()
        pt.positions = joint_positions
        sec  = int(duration)
        nsec = int((duration-sec)*1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal.trajectory.points = [pt]
        self._client.wait_for_server()
        future = self._client.send_goal_async(goal, feedback_callback=self.feedback)
        future.add_done_callback(self.goal_response)

# function for feedback
    def feedback(self, feedback):
        self.get_logger().debug(f"Feedback: {feedback}")

#function for goal response
    def goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn("Goal rejected")
            return
        gh.get_result_async().add_done_callback(self.result)

# function for result
    def result(self, future):
        res = future.result().result
        self.get_logger().info(f"Result code: {res.error_code}")
        rclpy.shutdown()

# function for plan motion linear
    def plan_motion_linear(
        self,
        goal_pose: List[float],
        start_cartesian_pose: Union[List[float], None] = None,
        start_joint_states: Union[List[float], None]    = None,
        speed: Optional[float] = None,
        acc:   Optional[float] = None,
        reusable: Optional[bool] = False,
    ) -> Tuple[Tuple[bool,bool], int, List[float]]:

        if not all([start_cartesian_pose, start_joint_states]):
            start_cartesian_pose = self.get_current_cartesian_pose()
            start_joint_states   = self.get_current_state_positions()


        assert len(goal_pose)==6, "goal_pose must be 6-vector"
        assert len(start_cartesian_pose)==6, "start_pose must be 6-vector"
        assert len(start_joint_states)==len(self.urdf_handler.actuated_indices)

        linear_prop = {
            "target_pose": [start_cartesian_pose, goal_pose],
            "speed":        self.speed_move_linear if speed is None else speed,
            "acceleration": self.acc_move_linear   if acc   is None else acc,
            "blending":     False,
            "blend_radius": 0.0,
        }

        plan_id = self._id_manager.update_id()
        cmd     = self._program.cmd.Linear

        self._program.set_command(
            cmd,
            **linear_prop,
            cmd_id=plan_id,
            current_joint_angles=start_joint_states,
            reusable_id=1 if reusable else 0
        )

        success_flags = self._id_manager.is_id_successful(plan_id)
        last_joint_state = None
        if all(success_flags):
            last_joint_state = self._program.get_last_joint_configuration(plan_id)

        return success_flags, plan_id, last_joint_state

# function for getting current state positions
    def get_current_state_positions(self) -> List[float]:
        return list(self._current_state.position)

# function for getting cartesain pose
    def get_current_cartesian_pose(self) -> List[float]:
        if self._current_state is None:
            raise RuntimeError("No joint state yet")
        full = np.zeros(len(self.urdf_handler.chain.links))
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            full[idx] = self._current_state.position[i]
        T = self.urdf_handler.chain.forward_kinematics(full)
        xyz = T[:3,3].tolist()
        rpy = R.from_matrix(T[:3,:3]).as_euler('xyz', degrees=False).tolist()
        return xyz + rpy

# main function
def main(args=None):
    rclpy.init(args=args)

    program    = PlannerProgram()
    id_manager = IDManager()

    urdf_path = os.path.expanduser(
        '~/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf'
    )
    node = MairaKinematics(urdf_path, program, id_manager)


    while rclpy.ok() and node._current_state is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    goal_pose = [-0.5, -0.4, 1.0, 0.0, 0.0, 0.0]
    success_flags, plan_id, last_joints = node.plan_motion_linear(goal_pose)
    if all(success_flags) and last_joints:
        joint_names = [f"joint{i+1}" for i in range(len(last_joints))]
        traj        = normalize_to_pi(np.array(last_joints)).tolist()
        node.send_joint_trajectory(joint_names, traj, duration=5.0)
        rclpy.spin(node)

# calling main function
if __name__ == '__main__':
    main()

####################################################################
JOINT --> CARTESIAN
#####################################################################

import os # imported os
import numpy as np # imported numpy

import rclpy #imported rclpy
from rclpy.node import Node  # imported Node
from rclpy.action import ActionClient # imported Actionclient

from sensor_msgs.msg import JointState # imported Joinstate
from trajectory_msgs.msg import JointTrajectoryPoint # imported JointTrajectory
from control_msgs.action import FollowJointTrajectory # imported FollowjointTrajecotry
from builtin_interfaces.msg import Duration # imported duration

from ikpy.chain import Chain # imported ik

# Normalize angle array into [-pi, pi)
def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi

# class URDF chain handler
class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        # Load full kinematic chain
        self.chain = Chain.from_urdf_file(urdf_path, base_elements=[base_link])
        # Indices 2..8 correspond to the 7 actuated joints
        self.actuated_indices = list(range(2, 9))
        # Deactivate fixed links to avoid IK warnings
        mask = [False] * len(self.chain.links)
        for idx in self.actuated_indices:
            mask[idx] = True
        self.chain.active_links_mask = mask

# function for inverse kinematics
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

# class Maira Kinematics
class MairaKinematics(Node):
    def __init__(self, urdf_path: str):
        super().__init__('maira_kinematics')
        self.urdf_handler = URDFChainHandler(urdf_path)

        # Trajectory action client
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        # Cache joint_state
        self._current_state: JointState = None
        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state,
            10
        )
#function for join state
    def joint_state(self, msg: JointState):
        # Cache the first received joint state as initial guess for IK
        self._current_state = msg
        self.destroy_subscription(self._js_sub)

# function for JOINT TO CARTESIAN
    def move_joint_to_cartesian(
        self,
        goal_pose: list[float],
        duration: float = 5.0
    ) -> bool:
        """
        Compute joint angles for Cartesian goal [x, y, z] and send a single-point trajectory.
        """
        if self._current_state is None:
            self.get_logger().error("No joint state received yet.")
            return False

        # Prepare initial full joint vector for IK
        initial_full = np.zeros(len(self.urdf_handler.chain.links))
        # Seed actuated indices with current positions
        for idx, link_idx in enumerate(self.urdf_handler.actuated_indices):
            joint_name = f"joint{idx+1}"
            if joint_name in self._current_state.name:
                js_index = self._current_state.name.index(joint_name)
                initial_full[link_idx] = self._current_state.position[js_index]

        # Compute IK
        target = np.array(goal_pose)
        full_solution = self.urdf_handler.inverse_kinematics(target, initial_joints=initial_full)

        # Extract actuated joint angles and normalize
        joint_angles = full_solution[self.urdf_handler.actuated_indices]
        joint_angles = normalize_to_pi(joint_angles)

        # Build trajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [f"joint{i+1}" for i in range(len(joint_angles))]
        pt = JointTrajectoryPoint()
        pt.positions = joint_angles.tolist()
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [pt]

        # Send goal
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available.")
            return False
        send = self._client.send_goal_async(goal)
        send.add_done_callback(lambda fut: None)
        return True

# main function
def main(args=None):
    rclpy.init(args=args)
    urdf = os.path.expanduser(
        '~/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf'
    )
    node = MairaKinematics(urdf)

    while rclpy.ok() and node._current_state is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    cart_goal = [0.5, 0.0, 0.3]
    if node.move_joint_to_cartesian(cart_goal, duration=4.0):
        node.get_logger().info("Sent Cartesian move goal.")
    else:
        node.get_logger().error("Failed to send goal.")

    rclpy.spin(node)

# calling main function
if __name__ == '__main__':
    main()


####################################################################
#  GET ELBOW UP IK SOLUTION
#####################################################################


import os  # imported os module
from copy import deepcopy # imported deepcopy module

import numpy as np # imported numpy module

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.action import ActionClient # imported Action client

from sensor_msgs.msg import JointState # imported Jointstate
from trajectory_msgs.msg import JointTrajectoryPoint # imported Jointrajectory
from control_msgs.action import FollowJointTrajectory # imported Followjoin trajectory
from builtin_interfaces.msg import Duration # imported Duration

from ikpy.chain import Chain # imported ik

# function for normalize to pi
def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    return (angles + np.pi) % (2.0 * np.pi) - np.pi

# class for URDF chain handler
class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.chain = Chain.from_urdf_file(urdf_path, base_elements=[base_link])
        # map chain links indices 2..8 to your 7 actuated joints
        self.actuated_indices = list(range(2, 9))

        # DEBUG: show what links we have
        print("URDFChainHandler:")
        print("  links:", [l.name for l in self.chain.links])
        print("  actuated_indices:", self.actuated_indices)

# function  for inverse kinematics
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        initial_position: np.ndarray
    ) -> np.ndarray:
        # delegate to ikpy
        return self.chain.inverse_kinematics(
            target_position,
            initial_position=initial_position
        )

# class MairaKinematics
class MairaKinematics(Node):
    def __init__(
        self,
        urdf_path: str,
        base_link: str = "maira7M_root_link",
        root_frame: str = "base_link",
    ):
        super().__init__('maira_kinematics')

        # IK handler
        self.urdf_handler = URDFChainHandler(urdf_path, base_link)

        # Trajectory action client
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        # to store first JointState
        self._current_state: JointState = None
        self._js_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state, 10
        )

# function for joint state
    def joint_state(self, msg: JointState):
        # grab the very first joint state
        self._current_state = msg
        self.destroy_subscription(self._js_sub)
        self.get_logger().info("Received initial joint state.")

# function for get elbow up ik solution
    def get_elbow_up_ik_solution(
        self,
        target_pos: np.ndarray,
    ) -> np.ndarray:
        if self._current_state is None:
            raise RuntimeError("No joint state received yet.")

        n_links = len(self.urdf_handler.chain.links)
        base_seed = np.zeros(n_links)

        # seed actuated joints with current positions
        for i, idx in enumerate(self.urdf_handler.actuated_indices):
            base_seed[idx] = self._current_state.position[i]

        # predefined elbow‐offset seeds: (d6, d7) = 0 or pi
        offsets = [(0.0, 0.0), (0.0, np.pi), (np.pi, 0.0), (np.pi, np.pi)]
        first_valid = None

        # Try the fixed offsets first
        for d6, d7 in offsets:
            seed = base_seed.copy()
            # the 5th/6th actuated joints are indices 6/7 in the full chain
            idx6 = self.urdf_handler.actuated_indices[4]
            idx7 = self.urdf_handler.actuated_indices[5]
            sign6 = (seed[idx6] < 0) * 2 - 1
            sign7 = (seed[idx7] < 0) * 2 - 1
            seed[idx6] += sign6 * d6
            seed[idx7] += sign7 * d7

            try:
                sol = self.urdf_handler.inverse_kinematics(target_pos, seed)
                self.get_logger().info(f"IK succeeded with seed offsets ({d6:.1f},{d7:.1f})")
                # You could insert your elbow‐up check here:
                # if self._elbow_checker.is_up(sol[self.urdf_handler.actuated_indices]):
                #     return sol
                if first_valid is None:
                    first_valid = sol
            except Exception as e:
                self.get_logger().warn(f"IK failed for seed offsets ({d6:.1f},{d7:.1f}): {e}")

        # Random-restart seeds
        for trial in range(10):
            seed = base_seed.copy()
            # perturb all actuated joints randomly in [-pi, +pi]
            seed[self.urdf_handler.actuated_indices] += np.random.uniform(-np.pi, np.pi, size=7)
            try:
                sol = self.urdf_handler.inverse_kinematics(target_pos, seed)
                self.get_logger().info(f"IK succeeded on random restart #{trial}")
                return sol
            except Exception as e:
                self.get_logger().debug(f"Random restart {trial} failed: {e}")

        # fallback to the first valid solution found
        if first_valid is not None:
            self.get_logger().warn("Falling back to first valid IK solution")
            return first_valid

        raise ValueError("No IK solution found after all seeds.")

# function for sending joint trajectory
    def send_joint_trajectory(self, joint_names: list[str], joint_positions: list[float], duration: float):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal.trajectory.points = [point]

        self._client.wait_for_server()
        future = self._client.send_goal_async(goal, feedback_callback=self.feedback)
        future.add_done_callback(self.goal_response)

# function for feedback
    def feedback(self, feedback):
        self.get_logger().debug(f"Action feedback: {feedback}")


#function for goal response
    def goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn("Trajectory goal rejected by server.")
            return
        gh.get_result_async().add_done_callback(self.result)

# function for result
    def result(self, future):
        res = future.result().result
        self.get_logger().info(f"Trajectory execution finished with code: {res.error_code}")
        rclpy.shutdown()

# main function
def main(args=None):
    rclpy.init(args=args)

    urdf_path = os.path.expanduser(
        '/home/midhun.eldose/neura/sim_ws/src/neura_robot_description/'
        'maira7M/urdf/maira7M.urdf'
    )
    node = MairaKinematics(urdf_path)

    # wait for the first joint state
    while rclpy.ok() and node._current_state is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    # pick a reachable target to debug first!
    target_pos = np.array([0.3, 0.0, 0.2])
    sol_full = node.get_elbow_up_ik_solution(target_pos)

    actuated = [sol_full[i] for i in node.urdf_handler.actuated_indices]
    actuated = normalize_to_pi(np.array(actuated)).tolist()
    joint_names = [f"joint{i+1}" for i in range(7)]

    node.send_joint_trajectory(joint_names, actuated, duration=5.0)
    rclpy.spin(node)

# calling main function
if __name__ == '__main__':
    main()



###################################################################
#  PLAN MOTION LINEAR VIA POINTS
####################################################################


import time  # imported time
from typing import List, Optional, Tuple, Union # imported List

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.action import ActionClient # imported Action client

from sensor_msgs.msg import JointState # imported jointstate
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # imported JointTrajectory point
from builtin_interfaces.msg import Duration # imported Duration
from control_msgs.action import FollowJointTrajectory # imported followjoint trajectory

# class Mairakinematics
class MairaKinematics(Node):
    def __init__(self):
        super().__init__("maira_kinematics")

        # Action client for trajectory execution
        self._traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_position_controller/follow_joint_trajectory"
        )

        # Current joint state cache
        self._current_joint_state: Optional[JointState] = None
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        # Joint names for the robot
        self._joint_names = [f"joint{i}" for i in range(1, 8)]

        # For planning
        # self._id_manager = IDManager()
        # self._program = ProgramBuilder()
        # Default speed/acc values
        self.speed_move_linear = 0.2   # m/s
        self.acc_move_linear = 0.5     # m/s²

        self._last_error_code: Optional[int] = None
        self._waiting_for_result = False


    # Joint state callback
    def joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg


    # Original motion execution helper
    def move_linear_via_points(
        self,
        positions_list: List[List[float]],
        times_list: List[float],
        speed_scale: Optional[float] = None,
    ) -> bool:
        """
        Send a JointTrajectory through FollowJointTrajectory action.

        positions_list: list of [q1,...,q7]
        times_list:     list of times (sec) for each point
        speed_scale:    optional factor to compress (>1) or expand (<1) timing
        """
        if len(positions_list) != len(times_list):
            raise ValueError("positions_list and times_list must match in length.")
        for pos in positions_list:
            if len(pos) != len(self._joint_names):
                raise ValueError(
                    f"Each point needs {len(self._joint_names)} joints, got {len(pos)}"
                )

        if not self._traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory action server not available.")
            return False

        # Build goal
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self._joint_names

        prev_pos = None
        prev_time = None

        for t_raw, positions in zip(times_list, positions_list):
            t = (t_raw / speed_scale) if speed_scale else t_raw
            sec = int(t)
            nsec = int((t - sec) * 1e9)

            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = Duration(sec=sec, nanosec=nsec)

            if prev_pos is not None and prev_time is not None:
                dt = max(t - prev_time, 1e-6)
                vel = [(p - pp) / dt for p, pp in zip(positions, prev_pos)]
            else:
                vel = [0.0] * len(positions)

            point.velocities = vel
            traj.points.append(point)
            prev_pos = positions
            prev_time = t

        goal.trajectory = traj
        goal.path_tolerance = []
        goal.goal_tolerance = []
        goal.goal_time_tolerance = Duration(sec=0, nanosec=0)

        send_future = self._traj_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)

        self._waiting_for_result = True
        start = self.get_clock().now()
        timeout_ns = int(10 * 1e9)
        while rclpy.ok() and self._waiting_for_result:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.get_clock().now() - start).nanoseconds > timeout_ns:
                self.get_logger().error("Timeout waiting for trajectory result.")
                self._waiting_for_result = False
                break

        if self._last_error_code == FollowJointTrajectory.Result().SUCCESSFUL:
            return True
        else:
            raise RuntimeError(
                f"Trajectory execution failed (error_code={self._last_error_code})"
            )

# function for goal response callback
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by server.")
            self._last_error_code = -1
            self._waiting_for_result = False
            return
        self.get_logger().info("Goal accepted, awaiting result...")
        goal_handle.get_result_async().add_done_callback(self.result_callback)

# function for feedback callback
    def feedback_callback(self, feedback):
        self.get_logger().debug(f"Feedback: {feedback}")

# function for result callback
    def result_callback(self, future):
        result = future.result().result
        self._last_error_code = result.error_code
        self.get_logger().info(f"Result received: error_code={self._last_error_code}")
        self._waiting_for_result = False


# function for motion linea via points
    def plan_motion_linear_via_points(
        self,
        goal_poses: List[List[float]],
        start_cartesian_pose: Union[List[float], None] = None,
        start_joint_states: Union[List[float], None] = None,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        rot_speed: Optional[float] = None,
        rot_acc: Optional[float] = None,
        blending_radius: Optional[float] = None,
        reusable: Optional[bool] = False,
    ) -> Tuple[Tuple[bool, bool], int, List[float]]:
        """
        Plan motion for a linear move-via-points sequence.

        Returns:
            (plan_successful, feasible), plan_id, last_joint_positions
        """
        # If neither start pose nor joints given, grab current
        if start_cartesian_pose is None and start_joint_states is None:
            start_cartesian_pose = self.get_current_cartesian_pose()
            start_joint_states = self.get_current_joint_state()

        # Validate inputs
        self.throw_if_list_poses_invalid(goal_poses)
        self.throw_if_pose_invalid(start_cartesian_pose)
        self.throw_if_joint_invalid(start_joint_states)

        # Prepend current pose to the goal list
        full_targets = [self.get_current_cartesian_pose()] + goal_poses

        linear_props = {
            "target_pose":  full_targets,
            "speed":        self.speed_move_linear if speed is None else speed,
            "acceleration": self.acc_move_linear if acc is None else acc,
            "blend_radius": 0.01 if blending_radius is None else blending_radius,
            # If your cmd.Linear supports rotation:
            # "rot_speed":    self.speed_move_rot if rot_speed is None else rot_speed,
            # "rot_acc":      self.acc_move_rot if rot_acc is None else rot_acc,
        }

        # Reserve a new plan ID
        plan_id = self._id_manager.update_id()

        # Insert command into the program
        self._program.set_command(
            cmd.Linear,
            **linear_props,
            cmd_id=plan_id,
            current_joint_angles=start_joint_states,
            reusable_id=1 if reusable else 0,
        )

        # Check planning success & feasibility
        success_flags = self.is_id_successful(plan_id)

        # Retrieve last joint configuration if successful
        last_joints: List[float] = []
        if all(success_flags):
            last_joints = self._program.get_last_joint_configuration(plan_id)

        return success_flags, plan_id, last_joints


    # Placeholder helpers (implement these as needed)
    def get_current_cartesian_pose(self) -> List[float]:
        # Query your kinematics or TF to get current end-effector pose
        raise NotImplementedError

# function for getitng curretn joint state
    def get_current_joint_state(self) -> List[float]:
        if self._current_joint_state is None:
            raise RuntimeError("No joint state received yet")
        return list(self._current_joint_state.position)

# function for throw if posese are invalid or not
    def throw_if_list_poses_invalid(self, poses: List[List[float]]):
        # Validate shape and values of each pose
        if not isinstance(poses, list) or not poses:
            raise TypeError("goal_poses must be a non-empty list of poses")

#function for checking if the pose is invalid or not
    def throw_if_pose_invalid(self, pose: List[float]):
        if not isinstance(pose, list) or len(pose) not in (6, 7):
            raise TypeError("Each pose must be a list of 6 or 7 floats")

# function for checking if joint is invalid or not
    def throw_if_joint_invalid(self, joints: List[float]):
        if not isinstance(joints, list) or len(joints) != len(self._joint_names):
            raise TypeError(f"Joint state must have {len(self._joint_names)} values")

# checking if id is succesful
    def is_id_successful(self, plan_id: int) -> Tuple[bool, bool]:
        # Query your planner/executor for flags (success, feasible)
        # For example:
        # return self._program.check_status(plan_id)
        raise NotImplementedError

# main function
def main(args=None):
    rclpy.init(args=args)
    node = MairaKinematics()

    # Example usage of move_linear_via_points:
    positions = [
        [0.0, 0.5, 0.3, -0.5, 0.0, 0.2, 0.0],
        [0.3, 1.0, 0.4, -0.2, 0.2, 0.3, 0.4],
    ]
    times = [2.0, 4.0]
    try:
        if node.move_linear_via_points(positions, times, speed_scale=1.5):
            node.get_logger().info("Motion completed successfully.")
    except Exception as e:
        node.get_logger().error(str(e))

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# calling main function
if __name__ == "__main__":
    main()

####################################################################
#  TRAJECTORY IS INVALID OR NOT
#####################################################################

import time # import time
from typing import List, Optional # import List, Optional

import rclpy # import rclpy
from rclpy.node import Node #import Node
from rclpy.action import ActionClient #import Action client
from trajectory_msgs.msg import JointTrajectoryPoint # import JointTrajectory
from sensor_msgs.msg import JointState #import Jointstate
from control_msgs.action import FollowJointTrajectory #import FollowJointTrajectory
from builtin_interfaces.msg import Duration # import Duration

# class Mairakinematics
class MairaKinematics(Node):

    def __init__(self):
        super().__init__("maira_kinematics")

        # Action client for FollowJointTrajectory
        action_name = "/joint_trajectory_position_controller/follow_joint_trajectory"
        self._client = ActionClient(self, FollowJointTrajectory, action_name)

        # Subscriber to /joint_states
        self._current_joint_state: Optional[JointState] = None
        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Number of joints (set after first state)
        self.num_joints: Optional[int] = None

# function for joint state callback
    def joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg
        # update number of joints on first callback
        if self.num_joints is None:
            self.num_joints = len(msg.name)

# function for throw if trajectory is invalid or not
    def _throw_if_trajectory_invalid(
        self, trajectory: List[List[float]]
    ) -> None:
        """Throw error if given trajectory is not valid.

        Parameters
        ----------
        trajectory : List[List[float]]
            Joint trajectory

        """
        if not isinstance(trajectory, list):
            raise TypeError(
                f"[ERROR] trajectory should be a List[List[float]] of length {self.num_joints}!"
            )
        for joint_states in trajectory:
            self.throw_if_joint_invalid(joint_states)

# function if the joint is invalid or not
    def throw_if_joint_invalid(self, joint_states: List[float]) -> None:
        """Throw error if a single waypoint is invalid."""
        if not isinstance(joint_states, list):
            raise TypeError(
                f"[ERROR] each waypoint must be a List[float] of length {self.num_joints}!"
            )
        if len(joint_states) != self.num_joints:
            raise ValueError(
                f"[ERROR] each waypoint must have {self.num_joints} values, got {len(joint_states)}."
            )
# function for trajecotry is valid
    def is_trajectory_valid(self, trajectory: List[List[float]]) -> bool:
        """
        Check whether the given trajectory is valid.
        Returns True if valid, False otherwise, logging any errors.
        """
        try:
            self._throw_if_trajectory_invalid(trajectory)
            return True
        except (TypeError, ValueError) as e:
            self.get_logger().error(f"Invalid trajectory: {e}")
            return False

# function for move joint to joint
    def move_joint_to_joint(
        self,
        goal_pose: List[float],
        speed: Optional[float] = None,
        acc: Optional[float] = None,
    ) -> bool:
        """
        Send the robot from its current joint state to the specified goal_pose.

        :param goal_pose: list of desired joint positions
        :param speed: (unused) placeholder for future velocity scaling
        :param acc: (unused) placeholder for future acceleration scaling
        :return: True if the goal was accepted, False otherwise
        """
        if self._current_joint_state is None:
            self.get_logger().warn('No current joint state available.')
            return False

        # Ensure we know how many joints to expect
        self.num_joints = len(self._current_joint_state.name)

        # Validate input trajectory (single-point list)
        if not self.is_trajectory_valid([goal_pose]):
            return False

        goal_msg = FollowJointTrajectory.Goal()
        # reuse existing joint ordering
        goal_msg.trajectory.joint_names = list(self._current_joint_state.name)

        # timestamp
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        # single waypoint
        point = JointTrajectoryPoint()
        point.positions = goal_pose[:]  # use passed-in pose
        # fixed duration of 1.0s for now (could be computed from speed/acc)
        point.time_from_start = Duration(sec=1, nanosec=0)

        goal_msg.trajectory.points = [point]

        self.get_logger().info('Waiting for action server...')
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False

        self.get_logger().info(f'Sending joint trajectory goal: {goal_pose}')
        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)
        return True

# function for goal response callback
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by the server.')
            return

        self.get_logger().info('Goal accepted → waiting for result…')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

# function for fedback callback
    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')


# function for getting result callback
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result received: error_code = {result.error_code}')
        rclpy.shutdown()

# main function
def main(args=None):
    rclpy.init(args=args)
    node = MairaKinematics()

    # wait up to 2s for first joint state
    start = node.get_clock().now()
    while rclpy.ok() and node._current_joint_state is None:
        elapsed = node.get_clock().now().nanoseconds - start.nanoseconds
        if elapsed > 2e9:
            node.get_logger().warn("No /joint_states received within 2 seconds")
            break
        rclpy.spin_once(node, timeout_sec=0.1)


    target_pose = [1.0, 2.0, 3.0, 4.0, 2.0, 5.0, 1.0]
    success = node.move_joint_to_joint(target_pose, speed=0.3, acc=0.2)
    if not success:
        node.get_logger().error('Failed to send trajectory goal.')

    rclpy.spin(node)
    rclpy.shutdown()

# calling main function
if __name__ == "__main__":
    main()

####################################################################
#  GET CURRENT CARTESIAN POSE (NOT GETTING RESULT)
#####################################################################

import time  # imported os module
from typing import List, Optional # imported List , Optional

import rclpy # imported rclpy
from rclpy.node import Node # imported Ndoe
from rclpy.action import ActionClient # imported Action client

from trajectory_msgs.msg import JointTrajectoryPoint # imported Joint Trajector point
from sensor_msgs.msg import JointState # imported Join state
from control_msgs.action import FollowJointTrajectory # imported Followjoin trajectory
from builtin_interfaces.msg import Duration as MsgDuration # imported Duration

import tf2_ros # imported tf2_ros
from geometry_msgs.msg import TransformStamped # imported Transform stamped

try:
    from tf_transformations import euler_from_quaternion
except ImportError:
    from scipy.spatial.transform import Rotation as R
    def euler_from_quaternion(quat: List[float]):
        r = R.from_quat(quat)
        return tuple(r.as_euler('xyz'))

# class Mairakinematics
class MairaKinematics(Node):
    def __init__(self, robot_state_client=None):
        super().__init__('maira_kinematics')
        self._robot_state = robot_state_client


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Action client for FollowJointTrajectory
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        # Subscribe to /joint_states
        self._current_joint_state: Optional[JointState] = None
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )


        self.create_timer(0.1, self.timer_log_tcp)

# function for joint state callback
    def joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg

# function for getting current joint state
    def get_current_joint_state(self) -> List[float]:
        """Return current joint positions via robot_state or /joint_states."""
        if self._robot_state is not None:
            try:
                return self._robot_state.getRobotStatus("jointAngles")
            except Exception as e:
                self.get_logger().warn(f"robot_state interface error: {e}")
        if self._current_joint_state is None:
            self.get_logger().warn('No current joint state available.')
            return []
        return list(self._current_joint_state.position)

# function for getting current cartesian pose
    def get_current_cartesian_pose(self) -> List[float]:
        """
        Lookup end‐effector pose in 'maira7M_root_link' → 'ee_link' using TF2.
        Returns [x, y, z, roll, pitch, yaw] or [] on failure.
        """
        try:
            now = rclpy.time.Time()
            t: TransformStamped = self.tf_buffer.lookup_transform(
                'maira7M_root_link',
                'ee_link',
                now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF2 lookup failed: {e}')
            return []

        x = t.transform.translation.x
        y = t.transform.translation.y
        z = t.transform.translation.z
        q = t.transform.rotation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return [x, y, z, roll, pitch, yaw]

# function for time for tcp
    def timer_log_tcp(self):
        pose6d = self.get_current_cartesian_pose()
        if pose6d:
            x, y, z, roll, pitch, yaw = pose6d
            self.get_logger().info(
                f'Live TCP → x={x:.3f}, y={y:.3f}, z={z:.3f}, '
                f'roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}'
            )
        else:
            self.get_logger().warn('Live TCP pose unavailable.')

# function for move joint to joint
    def move_joint_to_joint(
        self,
        goal_pose: List[float],
        speed: Optional[float] = None,
        acc: Optional[float] = None,
    ) -> bool:
        current = self.get_current_joint_state()
        if not current:
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = (
            list(self._current_joint_state.name)
            if self._current_joint_state else []
        )
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()

        start_pt = JointTrajectoryPoint(
            positions=current,
            time_from_start=MsgDuration(sec=0, nanosec=0)
        )
        end_pt = JointTrajectoryPoint(
            positions=goal_pose,
            time_from_start=MsgDuration(sec=1, nanosec=0)
        )
        goal.trajectory.points = [start_pt, end_pt]

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False

        self.get_logger().info(f'Sending goal → {goal_pose}')
        send_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)
        return True

# function for goal response callback
    def goal_response_callback(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Goal rejected')
            return
        self.get_logger().info('Goal accepted; awaiting result…')
        result_future = handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

# function for feedback callback
    def feedback_callback(self, feedback_msg):

        self.get_logger().info(f'Feedback → {feedback_msg.feedback}')


        pose6d = self.get_current_cartesian_pose()
        if pose6d:
            x, y, z, roll, pitch, yaw = pose6d
            self.get_logger().info(
                f'Live TCP on feedback → x={x:.3f}, y={y:.3f}, z={z:.3f}, '
                f'roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}'
            )
        else:
            self.get_logger().warn('Live TCP pose unavailable.')

# function for result callback
    def get_result_callback(self, future):
        res = future.result().result
        self.get_logger().info(f'Result: error_code={res.error_code}')

        pose6d = self.get_current_cartesian_pose()
        if pose6d:
            x, y, z, roll, pitch, yaw = pose6d
            self.get_logger().info(
                f'Final TCP → x={x:.3f}, y={y:.3f}, z={z:.3f}, '
                f'roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}'
            )
        else:
            self.get_logger().warn('Could not lookup TCP pose post‐move.')

        rclpy.shutdown()

# main function
def main(args=None):
    rclpy.init(args=args)
    node = MairaKinematics(robot_state_client=None)


    start = node.get_clock().now()
    while rclpy.ok() and node._current_joint_state is None:
        elapsed = node.get_clock().now().nanoseconds - start.nanoseconds
        if elapsed > 2e9:
            node.get_logger().warn("No /joint_states in 2s")
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    target = [1.0, 3.0, 1.0, 2.0, 5.0, 5.0, 1.0]
    if not node.move_joint_to_joint(target, speed=0.1, acc=0.1):
        node.get_logger().error('Failed to send trajectory goal.')

    rclpy.spin(node)
    rclpy.shutdown()

# callin main function
if __name__ == "__main__":
    main()


####################################################################
#  GET CURRENT JOINT STATE
#####################################################################

import time  # imported time
from typing import List, Optional # imported Optional

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.action import ActionClient # imported Actionclient

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # imported JointTrjaectory
from sensor_msgs.msg import JointState # imported Join state
from control_msgs.action import FollowJointTrajectory # imported FollowJoint Trajectory
from builtin_interfaces.msg import Duration # imported Duration

# class Mairakinematics
class MairaKinematics(Node):

    def __init__(self, robot_state_client=None):
        super().__init__("maira_kinematics")

        # Optional robot-state interface (e.g., to get jointAngles)
        self._robot_state = robot_state_client

        # Action client for FollowJointTrajectory
        action_name = "/joint_trajectory_position_controller/follow_joint_trajectory"
        self._client = ActionClient(self, FollowJointTrajectory, action_name)

        # Subscriber to /joint_states for fallback and joint names
        self._current_joint_state: Optional[JointState] = None
        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

# function for joint state callback
    def joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg

# function for getting current joint state
    def get_current_joint_state(self) -> List[float]:
        """Return current joint positions.

        Prefer the robot_state interface if available; otherwise use the last received /joint_states.
        """
        if self._robot_state is not None:
            try:
                return self._robot_state.getRobotStatus("jointAngles")
            except Exception as e:
                self.get_logger().warn(f"robot_state interface error: {e}")

        if self._current_joint_state is None:
            self.get_logger().warn('No current joint state available.')
            return []

        return list(self._current_joint_state.position)

# function for goal response callback
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by the server.')
            return

        self.get_logger().info('Goal accepted → waiting for result…')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

# function for feedback callback
    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')


# fucntion for getting result callback
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result received: error_code = {result.error_code}')
        rclpy.shutdown()

# function for move joint tot joint
    def move_joint_to_joint(self,
                            target_positions: List[float],
                            speed: float = 0.1,
                            acc: float = 0.1) -> bool:
        """
        Send a one-point FollowJointTrajectory goal to move from the current
        positions to `target_positions` at roughly the given speed and acceleration.

        Returns True if the goal was sent, False otherwise.
        """
        # Ensure we have current joint-state
        if self._current_joint_state is None:
            self.get_logger().warn("Cannot move: no /joint_states received yet.")
            return False

        # Wait for the server
        if not self._client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error(
                "FollowJointTrajectory action server not available → aborting move."
            )
            return False

        # Build the trajectory message
        traj = JointTrajectory()
        traj.joint_names = list(self._current_joint_state.name)

        # Compute time based on the largest joint delta
        current_positions = self.get_current_joint_state()
        deltas = [
            abs(tgt - cur)
            for tgt, cur in zip(target_positions, current_positions)
        ]
        max_delta = max(deltas) if deltas else 0.0
        move_time = max_delta / speed if speed > 0 else 1.0
        move_time = max(move_time, 1.0)  # enforce minimum 1 sec

        point = JointTrajectoryPoint()
        point.positions = target_positions
        # Optional: set a uniform velocity & acceleration profile
        point.velocities = [speed] * len(target_positions)
        point.accelerations = [acc] * len(target_positions)
        point.time_from_start = Duration(
            sec=int(move_time),
            nanosec=int((move_time - int(move_time)) * 1e9)
        )

        traj.points = [point]

        # Create and send the action goal
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

        return True

# main function
def main(args=None):
    rclpy.init(args=args)
    robot_state_client = None  # Or your actual state client
    node = MairaKinematics(robot_state_client)

    # Wait up to 2s for the first joint state
    start = node.get_clock().now()
    while rclpy.ok() and node._current_joint_state is None:
        elapsed = node.get_clock().now().nanoseconds - start.nanoseconds
        if elapsed > 2e9:
            node.get_logger().warn("No /joint_states received within 2 seconds")
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    # Define your target joint positions here
    target_pose = [1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 1.0]
    success = node.move_joint_to_joint(target_pose, speed=0.5, acc=0.5)
    if not success:
        node.get_logger().error('Failed to send trajectory goal.')

    rclpy.spin(node)
    rclpy.shutdown()

# calling main function
if __name__ == "__main__":
    main()


###################################################################
 #MOVE JOINT_TO_JOINT
####################################################################

import time  # imported os
from typing import List, Optional # imported  List

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.action import ActionClient # imported Action client
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # imported JointTrajectory
from sensor_msgs.msg import JointState # imported joinstate
from control_msgs.action import FollowJointTrajectory # imported FollowjointTrajectory
from builtin_interfaces.msg import Duration # imported duration

# class Mairakinematics
class MairaKinematics(Node):

    def __init__(self):
        super().__init__("maira_kinematics")

        # Action client for FollowJointTrajectory
        action_name = "/joint_trajectory_position_controller/follow_joint_trajectory"
        self._client = ActionClient(self, FollowJointTrajectory, action_name)

        # Subscriber to /joint_states
        self._current_joint_state: Optional[JointState] = None
        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

#function for joint state callback
    def joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg

# function for move joint to joint
    def move_joint_to_joint(
        self,
        goal_pose: List[float],
        speed: Optional[int] = None,
        acc: Optional[int] = None,
    ) -> bool:
        """
        Send the robot from its current joint state to the specified goal_pose.

        :param goal_pose: list of desired joint positions
        :param speed: (unused) placeholder for future velocity scaling
        :param acc: (unused) placeholder for future acceleration scaling
        :return: True if the goal was accepted, False otherwise
        """
        if self._current_joint_state is None:
            self.get_logger().warn('No current joint state available.')
            return False

        goal_msg = FollowJointTrajectory.Goal()
        # reuse existing joint ordering
        goal_msg.trajectory.joint_names = list(self._current_joint_state.name)

        # timestamp
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        # single waypoint
        point = JointTrajectoryPoint()
        point.positions = goal_pose[:]  # use passed-in pose
        # fixed duration of 1.0s for now (could be computed from speed/acc)
        point.time_from_start = Duration(sec=1, nanosec=0)

        goal_msg.trajectory.points = [point]

        self.get_logger().info('Waiting for action server...')
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False

        self.get_logger().info(f'Sending joint trajectory goal: {goal_pose}')
        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)
        return True

# function for goal response callback
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by the server.')
            return

        self.get_logger().info('Goal accepted → waiting for result…')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

# function for feedback callback
    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

# function for result callback
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result received: error_code = {result.error_code}')
        rclpy.shutdown()

# main function
def main(args=None):
    rclpy.init(args=args)
    node = MairaKinematics()

    # wait up to 2s for first joint state
    start = node.get_clock().now()
    while rclpy.ok() and node._current_joint_state is None:
        elapsed = node.get_clock().now().nanoseconds - start.nanoseconds
        if elapsed > 2e9:
            node.get_logger().warn("No /joint_states received within 2 seconds")
            break
        rclpy.spin_once(node, timeout_sec=0.1)


    target_pose = [1.0, 4.0, 7.0, 4.0, 2.0, 5.0, 1.0] # setting target pose
    success = node.move_joint_to_joint(target_pose,speed=0.1,acc=0.1) # passing values to the move joint fucntion
    if not success:
        node.get_logger().error('Failed to send trajectory goal.')

    rclpy.spin(node)
    rclpy.shutdown()

#calling main function
if __name__=="__main__":
    main()

####################################################################
#  PLAN MOTION JOINT TO JOINT (TOLERANCE ISSUE)
#####################################################################

import time  # imported time module
from typing import List, Optional, Tuple  # imported typing

import rclpy  # ROS2 client library
from rclpy.node import Node  # Node base class
from rclpy.action import ActionClient  # Action client for sending goals
import rclpy.duration  # for creating a delayed timestamp

from sensor_msgs.msg import JointState  # current joint states
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # trajectory messages
from builtin_interfaces.msg import Duration as MsgDuration  # for goal/waypoint timing

from control_msgs.action import FollowJointTrajectory  # follow trajectory action
from control_msgs.msg import JointTolerance  # to specify tolerances

# class Mairakinematics
class MairaKinematics(Node):
    def __init__(self):
        super().__init__("maira_kinematics")

        # Action client for joint trajectory control
        self._traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_position_controller/follow_joint_trajectory"
        )

        # Subscribe to /joint_states for current state
        self._current_joint_state: Optional[JointState] = None
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        # 7 joints named joint1…joint7
        self._joint_names = [f"joint{i}" for i in range(1, 8)]
        self._last_error_code: Optional[int] = None
        self._waiting_for_result = False

        # Defaults (rad/sec and rad/sec^2)
        self.speed_move_joint = 0.2   # rad/sec (slower)
        self.acc_move_joint = 0.2     # rad/sec^2

        # Number of intermediate waypoints for smoother trajectory
        self.num_waypoints = 10

    # callback for joint states
    def joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg

    # callback when goal response is received
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by server.")
            self._last_error_code = -1
            self._waiting_for_result = False
            return

        self.get_logger().info("Goal accepted, awaiting result...")
        fut = goal_handle.get_result_async()
        fut.add_done_callback(self.result_callback)

    # feedback callback
    def feedback_callback(self, feedback):
        pos_err = [abs(d - a) for d, a in zip(feedback.desired.positions, feedback.actual.positions)]
        vel_err = [abs(d - a) for d, a in zip(feedback.desired.velocity, feedback.actual.velocity)]
        self.get_logger().info(f"Pos err: {pos_err}\nVel err: {vel_err}")

    # result callback
    def result_callback(self, future):
        result = future.result().result
        self._last_error_code = result.error_code
        self.get_logger().info(f"Result received: error_code={self._last_error_code}")
        self._waiting_for_result = False

    # plan and send joint-to-joint motion
    def plan_motion_joint_to_joint(
        self,
        goal_pose: List[float],
        start_joint_states: Optional[List[float]] = None,
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        reusable: Optional[bool] = False,
    ) -> Tuple[Tuple[bool, bool], int, List[float]]:
        # 1) starting state
        if start_joint_states is None:
            start_joint_states = self.get_current_joint_state()

        self.throw_if_joint_invalid(start_joint_states)
        self.throw_if_joint_invalid(goal_pose)

        # 2) wait for action server
        if not self._traj_client.server_is_ready():
            self.get_logger().info("Waiting for trajectory action server...")
            if not self._traj_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("Trajectory action server not available!")
                return (False, False), -1, []

        # 3) build trajectory
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self._joint_names

        # stamp the trajectory a bit into the future to avoid timestamp errors
        start_time = self.get_clock().now() + rclpy.duration.Duration(seconds=0.1)
        traj.header.stamp = start_time.to_msg()

        # Compute timing based on largest joint move
        max_delta = max(abs(g - s) for g, s in zip(goal_pose, start_joint_states))
        vel = speed if speed is not None else self.speed_move_joint
        dur = max_delta / vel if vel > 0 else 1.0

        # Create trajectory points
        points: List[JointTrajectoryPoint] = []

        # Start point
        p0 = JointTrajectoryPoint()
        p0.positions = list(start_joint_states)
        p0.velocities = [0.0] * len(start_joint_states)
        p0.time_from_start = MsgDuration(sec=0, nanosec=0)
        points.append(p0)

        # Intermediate waypoints
        prev_positions = start_joint_states
        prev_time = 0.0
        for i in range(1, self.num_waypoints):
            ratio = i / float(self.num_waypoints)
            t = dur * ratio
            pos = [s + ratio * (g - s) for s, g in zip(start_joint_states, goal_pose)]
            dt = max(t - prev_time, 1e-6)
            vel_list = [(p - pp) / dt for p, pp in zip(pos, prev_positions)]

            pi = JointTrajectoryPoint()
            pi.positions = pos
            pi.velocities = vel_list
            pi.time_from_start = MsgDuration(sec=int(t), nanosec=int((t % 1) * 1e9))
            points.append(pi)

            prev_positions = pos
            prev_time = t

        # End point with zero velocity
        p1 = JointTrajectoryPoint()
        p1.positions = list(goal_pose)
        p1.velocities = [0.0] * len(goal_pose)
        p1.time_from_start = MsgDuration(sec=int(dur), nanosec=int((dur % 1) * 1e9))
        points.append(p1)
        traj.points = points
        goal.trajectory = traj

        # 4) set reasonable tolerances
        tol = 0.05       # 50 mrad positional tolerance
        vel_tol = 0.1   # rad/s velocity tolerance
        acc_tol = 0.1   # rad/s^2 acceleration tolerance

        goal.path_tolerance = [
            JointTolerance(name=jn, position=tol, velocity=vel_tol, acceleration=acc_tol)
            for jn in self._joint_names
        ]
        goal.goal_tolerance = [
            JointTolerance(name=jn, position=tol/2, velocity=vel_tol, acceleration=acc_tol)
            for jn in self._joint_names
        ]
        # allow 1 second to settle at final goal
        goal.goal_time_tolerance = MsgDuration(sec=1, nanosec=0)

        # 5) send & wait
        self._waiting_for_result = True
        send_fut = self._traj_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        send_fut.add_done_callback(self.goal_response_callback)

        while self._waiting_for_result:
            rclpy.spin_once(self, timeout_sec=0.1)

        # 6) interpret result
        sent_ok = (self._last_error_code is not None) and (self._last_error_code != -1)
        exec_ok = (self._last_error_code == FollowJointTrajectory.Result().SUCCESSFUL)
        last_state = goal_pose if exec_ok else start_joint_states
        plan_id = int(self.get_clock().now().nanoseconds % 1_000_000)

        return (sent_ok, exec_ok), plan_id, last_state

    # get latest joint state
    def get_current_joint_state(self) -> List[float]:
        if self._current_joint_state is None:
            raise RuntimeError("No joint state received yet")
        return list(self._current_joint_state.position)

    # validate joint array length
    def throw_if_joint_invalid(self, joints: List[float]):
        if len(joints) != len(self._joint_names):
            raise TypeError(
                f"Expected {len(self._joint_names)} joint values, got {len(joints)}"
            )

# main function

def main(args=None):
    rclpy.init(args=args)
    node = MairaKinematics()
    try:
        node.get_logger().info("Waiting up to 5s for initial joint state...")
        deadline = time.time() + 5.0
        while node._current_joint_state is None and time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

        goal_positions = [1.0, -0.2, 1.0, -0.75, 0.3, 0.4, -0.2]
        node.get_logger().info(f"Sending goal: {goal_positions}")
        (sent, done), pid, final = node.plan_motion_joint_to_joint(
            goal_pose=goal_positions,
            speed=0.1,   # rad/sec (slower)
            acc=0.1      # rad/sec^2
        )
        node.get_logger().info(
            f"Sent: {sent}, Executed: {done}, ID={pid}, State={final}"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

# calling main function
if __name__ == "__main__":
    main()

####################################################################
#  MOVE LINEAR VIA POINTS
#####################################################################

import time # import time module
from typing import List, Optional # imported List, Optional

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.action import ActionClient # imported Actionclient

from sensor_msgs.msg import JointState  # imported Jointstate
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # imported JointTrajectory
from builtin_interfaces.msg import Duration # imported Duration
from control_msgs.action import FollowJointTrajectory # imported FollowJointTrajectory


# class MairaKinematics

class MairaKinematics(Node):
    def __init__(self):
        super().__init__("maira_kinematics")


        self._traj_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_position_controller/follow_joint_trajectory"
        )

        self._current_joint_state: Optional[JointState] = None
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

# setting the joint names
        self._joint_names = [f"joint{i}" for i in range(1, 8)]


        self._last_error_code: Optional[int] = None
        self._waiting_for_result = False

# function for joint state callback
    def joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg

# function for moving linear via points
    def move_linear_via_points(
        self,
        positions_list: List[List[float]],
        times_list: List[float],
        speed_scale: Optional[float] = None,
    ) -> bool:
        """
        Send a JointTrajectory through FollowJointTrajectory action.

        positions_list: list of [q1,...,q7]
        times_list: list of times (sec) for each point
        speed_scale: optional factor to compress (>1) or expand (<1) timing
        """


        if len(positions_list) != len(times_list):
            raise ValueError("positions_list and times_list must match in length.")
        for pos in positions_list:
            if len(pos) != len(self._joint_names):
                raise ValueError(
                    f"Each point needs {len(self._joint_names)} joints, got {len(pos)}"
                )


        if not self._traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory action server not available.")
            return False

# setting goal
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self._joint_names

        prev_pos = None
        prev_time = None

        for t_raw, positions in zip(times_list, positions_list):

            t = (t_raw / speed_scale) if speed_scale else t_raw
            sec = int(t)
            nsec = int((t - sec) * 1e9)

            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = Duration(sec=sec, nanosec=nsec)


            if prev_pos is not None and prev_time is not None:
                dt = max(t - prev_time, 1e-6)
                vel = [
                    (p - pp) / dt for p, pp in zip(positions, prev_pos)
                ]
            else:
                vel = [0.0] * len(positions)

            point.velocities = vel

            traj.points.append(point)
            prev_pos = positions
            prev_time = t

        goal.trajectory = traj

        goal.path_tolerance = []
        goal.goal_tolerance = []
        goal.goal_time_tolerance = Duration(sec=0, nanosec=0)


        send_future = self._traj_client.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)

        self._waiting_for_result = True
        start = self.get_clock().now()
        timeout_ns = int(10 * 1e9)
        while rclpy.ok() and self._waiting_for_result:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.get_clock().now() - start).nanoseconds > timeout_ns:
                self.get_logger().error("Timeout waiting for trajectory result.")
                self._waiting_for_result = False
                break


        if self._last_error_code == FollowJointTrajectory.Result().SUCCESSFUL:
            return True
        else:
            raise RuntimeError(
                f"Trajectory execution failed (error_code={self._last_error_code})"
            )

# function for goal response callback

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by server.")
            self._last_error_code = -1
            self._waiting_for_result = False
            return

        self.get_logger().info("Goal accepted, awaiting result...")
        goal_handle.get_result_async().add_done_callback(self.result_callback)

# function for feedback callback
    def feedback_callback(self, feedback):

        self.get_logger().debug(f"Feedback: {feedback}")

# function for result callback
    def result_callback(self, future):
        result = future.result().result
        self._last_error_code = result.error_code
        self.get_logger().info(f"Result received: error_code={self._last_error_code}")
        self._waiting_for_result = False

# creating main function
def main(args=None):
    rclpy.init(args=args)
    node = MairaKinematics()

# setting example positons
    positions = [
        [0.0, 0.5, 0.3, -0.5, 0.0, 0.2, 0.0],
        [0.0, 0.0, 0.1, -0.2, 0.4, 0.6, 0.8],
    ]
    times = [2.0, 4.0]

    try:
        success = node.move_linear_via_points(positions, times, speed_scale=1.5)
        if success:
            node.get_logger().info("Motion completed successfully.")
    except Exception as e:
        node.get_logger().error(str(e))

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# calling main function
if __name__ == "__main__":
    main()


####################################################################
#  MOVE JOINT VIA POINTS
#####################################################################


import time # imported time module
from typing import List, Optional # imported List

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.action import ActionClient # imported Actionclient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # imported JointTrajectory
from sensor_msgs.msg import JointState # imported joinstate
from control_msgs.action import FollowJointTrajectory # imported FollowJointtrajectory
from builtin_interfaces.msg import Duration # imported Duration

# class Mairakinematics
class MairaKinematics(Node):
    def __init__(self):
        super().__init__("maira_kinematics")

        # FollowJointTrajectory action client
        traj_action_name = "/joint_trajectory_position_controller/follow_joint_trajectory"
        self._traj_client = ActionClient(self, FollowJointTrajectory, traj_action_name)

        # Subscribe to joint states
        self._current_joint_state: Optional[JointState] = None
        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

        # Default joint names
        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # Internal flags
        self._last_error_code: Optional[int] = None
        self._waiting_for_result = False

# function for joint state callback
    def joint_state_callback(self, msg: JointState):
        self._current_joint_state = msg

# function for move joint via points
    def move_joint_via_points(
        self,
        positions_list: List[List[float]],
        times_list: List[float],
        speed_scale: Optional[float] = None,
        trajectory: List[List[float]]=None,
        speed: Optional[int] = None,
        acc: Optional[int] = None,
    ) -> bool:
        # Validate inputs
        if len(positions_list) != len(times_list):
            raise ValueError("positions_list and times_list must be the same length.")
        for pos in positions_list:
            if len(pos) != len(self._joint_names):
                raise ValueError(
                    f"Each position must have {len(self._joint_names)} entries, got {len(pos)}."
                )

        # Wait for server
        if not self._traj_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory action server not available.")
            return False

        # Build goal
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = self._joint_names

        for t, positions in zip(times_list, positions_list):
            point = JointTrajectoryPoint()
            point.positions = positions
            # Scale time if needed
            duration_val = t / speed_scale if speed_scale else t
            # Build ROS duration message
            seconds = int(duration_val)
            nanoseconds = int((duration_val - seconds) * 1e9)
            point.time_from_start = Duration(sec=seconds, nanosec=nanoseconds)
            traj.points.append(point)

        goal.trajectory = traj

        # Send goal
        send_goal_future = self._traj_client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback,
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

        # Wait for result
        self._waiting_for_result = True
        start = self.get_clock().now()
        timeout_ns = int(10 * 1e9)  # 10 seconds
        while rclpy.ok() and self._waiting_for_result:
            rclpy.spin_once(self, timeout_sec=0.1)
            elapsed_ns = (self.get_clock().now() - start).nanoseconds
            if elapsed_ns > timeout_ns:
                self.get_logger().error("Timeout waiting for trajectory result.")
                self._waiting_for_result = False
                break

        if self._last_error_code == 0:
            return True
        else:
            raise RuntimeError(
                f"Trajectory execution failed (error_code={self._last_error_code})"
            )

# function for goal reponse callback
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected by server.")
            self._last_error_code = -1
            self._waiting_for_result = False
            return

        self.get_logger().info("Goal accepted, awaiting result...")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.result_callback)

# function for feedback callback
    def feedback_callback(self, feedback):
        self.get_logger().debug(f"Feedback: {feedback}")

# function for result callback
    def result_callback(self, future):
        result = future.result().result
        self._last_error_code = result.error_code
        self.get_logger().info(f"Result received: error_code={self._last_error_code}")
        self._waiting_for_result = False

# main  function
def main(args=None):
    rclpy.init(args=args)
    node = MairaKinematics()

    # Example usage: move joints to two waypoints
    positions = [
        [0.0, 0.5, 0.3, -0.2, 0.0, 0.1, 0.0],
        [0.1, 0.6, 0.1, -0.4, 0.1, 0.0, 0.1],
    ]
    times = [2.0, 4.0]  # seconds

    try:
        success = node.move_joint_via_points(positions, times, speed_scale=1.5,speed=0.3,acc=0.3)
        if success:
            node.get_logger().info("Motion completed successfully.")
    except Exception as e:
        node.get_logger().error(str(e))

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# calling main function
if __name__=="__main__":
    main()

####################################################################
# Ik SOLVER (MOVE LINEAR)
#####################################################################

import sys # imported sys module
import os # imported os module
import time # imported time
from typing import List, Optional # imported List ,Optional

import numpy as np # imported numpy

import rclpy # imported rclpy
from rclpy.node import Node # imported Node
from rclpy.duration import Duration # imported Duration
from rclpy.action import ActionClient # imported Action client

from sensor_msgs.msg import JointState # imported Jointstate
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # imported Jointtrajectory
from control_msgs.action import FollowJointTrajectory # imported FollowjointTrajectory

from ikpy.chain import Chain # imported ik

# function to normalize pi
def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap each element of angles to the range [-π, π)."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


# class URDF chain handler
class URDFChainHandler:
    def __init__(self, urdf_path: str, base_link: str = "maira7M_root_link"):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.chain: Optional[Chain] = None

# function for load chain
    def load_chain(self):
        if not os.path.isfile(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        self.chain = Chain.from_urdf_file(
            self.urdf_path,
            base_elements=[self.base_link]
        )
        self.print_joint_indices()

# function for print joint indices
    def print_joint_indices(self):
        logger = rclpy.logging.get_logger('URDFChainHandler')
        logger.info("Hard coded joint indices:")
        for i in range(2, 9):
            logger.info(f"  {i}: maira7M_joint{i-1}")

#function for inverse kinematics
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        initial_joints: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.chain is None:
            raise RuntimeError("IK chain not loaded; call load_chain() first")

        n_links = len(self.chain.links)
        init_full = np.zeros(n_links)
        if initial_joints is not None:
            init_full[2:2 + len(initial_joints)] = initial_joints

        sol = self.chain.inverse_kinematics(
            target_position,
            initial_position=init_full
        )
        if sol is None:
            raise RuntimeError(f"No IK solution for target {target_position.tolist()}")
        return sol

# class Mairalinear
class MairaLinearDIY(Node):
    def __init__(self):
        super().__init__('maira_linear_diy')

        # Parameters
        self.declare_parameter('urdf_path', '/home/midhun.eldose/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf')
        self.declare_parameter('base_link', 'maira7M_root_link')
        urdf_path = self.get_parameter('urdf_path').get_parameter_value().string_value
        base_link = self.get_parameter('base_link').get_parameter_value().string_value

        # IK handler
        self.ik_handler = URDFChainHandler(urdf_path, base_link)
        self.ik_handler.load_chain()

        # Subscriber
        self.latest_joint_state: Optional[JointState] = None
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Action client
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        # Defaults
        self.speed_move_linear = 0.8
        self.acc_move_linear = 0.6

# function for joint state callback
    def joint_state_callback(self, msg: JointState):
        self.latest_joint_state = msg


# function for waitng for first joint state
    def wait_for_first_joint_state(self, timeout: float = 5.0):
        start = time.time()
        while self.latest_joint_state is None:
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for first joint state")
            self.get_logger().info("Waiting for /joint_states...")
            rclpy.spin_once(self, timeout_sec=0.1)

# function for getting current joint state
    def get_current_joint_state(self) -> List[float]:
        if self.latest_joint_state is None:
            self.wait_for_first_joint_state()
        return list(self.latest_joint_state.position)

#function for normalizing the quaternion
    def _normalize_quaternion(self, q: List[float]) -> List[float]:
        arr = np.array(q, dtype=float)
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise ValueError("Zero-length quaternion")
        return (arr / norm).tolist()

#function for ik solver
    def _ik_solver(self, current_js: List[float], goal_pose: List[float]) -> Optional[List[float]]:
        logger = self.get_logger()
        target_pos = np.array(goal_pose[:3])
        try:
            full_solution = self.ik_handler.inverse_kinematics(
                target_pos,
                initial_joints=np.array(current_js)
            )
        except RuntimeError as e:
            logger.info(f"IK success: False - {e}")
            return None

        actuated = full_solution[2:2+7]
        result = normalize_to_pi(np.array(actuated)).tolist()
        logger.info("IK success: True")
        return result

# function for move linear
    def move_linear(
        self,
        goal_pose: List[float],
        speed: Optional[float] = None,
        acc: Optional[float] = None,
        joint_states: Optional[List[str]] = None,
    ) -> bool:
        if not isinstance(goal_pose, list) or len(goal_pose) != 7:
            raise TypeError("Pose must be a list of 7 floats [x,y,z,qx,qy,qz,qw] or 7 joint angles if using joint_states")

        if joint_states is None:
            goal_pose[3:] = self._normalize_quaternion(goal_pose[3:])

        traj = JointTrajectory()
        point = JointTrajectoryPoint()

        if joint_states is not None:
            if len(joint_states) != len(goal_pose):
                raise ValueError("joint_states and goal_pose must match in length")
            traj.joint_names = joint_states
            point.positions = [float(v) for v in goal_pose]
            duration_sec = 2.0
        else:
            self.wait_for_first_joint_state()
            current_js = self.get_current_joint_state()
            target_js = self._ik_solver(current_js, goal_pose)
            if target_js is None:
                self.get_logger().error("Aborting move: IK failed")
                return False

            traj.joint_names = [link.name for link in self.ik_handler.chain.links[2:2+7]]
            point.positions = [float(v) for v in target_js]
            speed = speed or self.speed_move_linear
            distance = np.linalg.norm(np.array(goal_pose[:3]))
            duration_sec = distance / speed

        point.time_from_start = Duration(seconds=duration_sec).to_msg()
        traj.points = [point]

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False

        send_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by controller')
            return False

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result

        if result.error_code == 0:
            self.get_logger().info('Move succeeded.')
            self.get_logger().info('IK solver successful: robot moved to target pose')
            return True
        else:
            self.get_logger().error(f'Move failed with error code {result.error_code}')
            return False

# main function
def main(args=None):
    rclpy.init(args=args)
    node = MairaLinearDIY()

    try:
        joint_states = [f'joint{i}' for i in range(1,8)]
        goal_pose = [0.7, 0.5, -0.5, 0.0, 1.0, 0.2, 0.2]
        node.move_linear(goal_pose=goal_pose, speed=0.9, acc=0.3, joint_states=joint_states)
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Exception: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

# calling main function
if __name__ == '__main__':
    main()
