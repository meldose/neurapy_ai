# ### action/MoveToPose.action

# # Goal definition
# geometry_msgs/PoseStamped target_pose

# # Result definition
# bool success

# # Feedback definition
# float32 progress

# ### database_client.py

# """ROS2 client to read the robot database."""
# import rclpy
# from rclpy.node import Node
# from neura_ai_database_msgs.srv import (
#     ReadEndEffector,
#     ReadTCPPose,
#     ReadWorkspace,
# )
# from std_srvs.srv import Trigger

# from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
# from neurapy_ai.utils.ros_conversions import geometry_msg_pose_2_pose
# from neurapy_ai.utils.types import EndEffector, JointState, Pose, TCPPose, Workspace

# class DatabaseClient(Node):
#     """Client to read the robot database in ROS2."""

#     def __init__(self):
#         super().__init__('neura_ai_database_client')
#         # service clients
#         self._point_client = self.create_client(ReadTCPPose, '/neura_ai_database/read_tcpPose')
#         self._workspace_client = self.create_client(ReadWorkspace, '/neura_ai_database/read_workspace')
#         self._ee_client = self.create_client(ReadEndEffector, '/neura_ai_database/read_end_effector')
#         self._update_client = self.create_client(Trigger, '/neura_ai_database/update_database')
#         # wait for services
#         for client in [self._point_client, self._workspace_client, self._ee_client, self._update_client]:
#             if not client.wait_for_service(timeout_sec=5.0):
#                 self.get_logger().error(f"Service {client.srv_name} not available")
#         # update cache
#         self.update_database()

#     def get_pose(self, point_name: str):
#         req = ReadTCPPose.Request()
#         req.tcp_point_name = point_name
#         future = self._point_client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)
#         resp = future.result()
#         if resp is None:
#             return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED), None, None
#         if resp.return_code.value < 0:
#             return ReturnCode(resp.return_code.value, resp.return_code.message), None, None
#         pose = geometry_msg_pose_2_pose(resp.tcp_pose.transform_tcp2ref)
#         joints = list(resp.tcp_pose.tcp_pose_joint_space)
#         return ReturnCode(), pose, joints

#     def get_workspace(self, workspace_name: str):
#         req = ReadWorkspace.Request()
#         req.workspace_name = workspace_name
#         future = self._workspace_client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)
#         resp = future.result()
#         if resp.return_code.value < 0:
#             return ReturnCode(resp.return_code.value, resp.return_code.message), None
#         ws = resp.workspace
#         lookats = [
#             TCPPose(
#                 geometry_msg_pose_2_pose(point.transform_tcp2ref),
#                 JointState(point.tcp_pose_joint_space)
#             ) for point in ws.lookat_points
#         ]
#         workspace = Workspace(
#             pose=geometry_msg_pose_2_pose(ws.transform_workspace2ref),
#             frame=ws.ref_frame,
#             len_x=ws.x_max,
#             len_y=ws.y_max,
#             len_z=ws.z_max,
#             lookat_poses=lookats,
#             name=workspace_name,
#             type="tabletop" if ws.type == ws.TABLETOP else "bin",
#             mesh_model=ws.mesh_model,
#             collision_padding=ws.collision_padding,
#         )
#         return ReturnCode(), workspace

#     def get_end_effector(self, ee_name: str=''):
#         req = ReadEndEffector.Request()
#         req.end_effector_name = ee_name
#         future = self._ee_client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)
#         resp = future.result()
#         if resp.return_code.value < 0:
#             return ReturnCode(resp.return_code.value, resp.return_code.message), None
#         tcp_pose = geometry_msg_pose_2_pose(resp.end_effector.transform_tcp2flange)
#         ee = EndEffector(
#             name=resp.end_effector_name,
#             neura_supported_typename=resp.neura_supported_typename,
#             tcp_pose=tcp_pose,
#         )
#         return ReturnCode(), ee

#     def update_database(self):
#         req = Trigger.Request()
#         future = self._update_client.call_async(req)
#         rclpy.spin_until_future_complete(self, future)
#         resp = future.result()
#         if not resp.success:
#             self.get_logger().error(f"Update failed: {resp.message}")
#             return ReturnCode(False, resp.message)
#         return ReturnCode(True, resp.message)

### maira_kinematics.py

import threading
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String, Int32MultiArray, Float32MultiArray,Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
import numpy as np
from copy import deepcopy

from neura_apps.gui_program.program import Program
from neurapy.commands.state.robot_status import RobotStatus
from neurapy.robot import Robot
from neurapy.state_flag import cmd
from neurapy.utils import CmdIDManager
from database_client import DatabaseClient
# from neurapy_ai.utils.types import Pose as AiPose, EndEffector
# from neurapy_ai_utils.robot.elbow_checker import ElbowChecker

class ThreadSafeCmdIDManager:
    def __init__(self, id_manager=None):
        self._lock = threading.Lock()
        self._mgr = id_manager or CmdIDManager()
    def update_id(self) -> int:
        with self._lock:
            return self._mgr.update_id()

class MairaKinematics(Node):
    def __init__(
        self,
        speed_move_joint: int = 20,
        speed_move_linear: float = 0.1,
        rot_speed_move_linear: float = 0.87266463,
        acc_move_joint: int = 20,
        acc_move_linear: float = 0.1,
        rot_acc_move_linear: float = 1.74532925,
        blending_radius: float = 0.005,
        require_elbow_up: bool = True,
        id_manager: CmdIDManager = None,
        robot_handler: Robot = None,
    ):
        super().__init__('maira_kinematics')
        # parameters
        self.speed_move_joint = speed_move_joint
        self.acc_move_joint = acc_move_joint
        self.speed_move_linear = speed_move_linear
        self.acc_move_linear = acc_move_linear
        self.require_elbow_up = require_elbow_up
        # id manager
        self._id_manager = ThreadSafeCmdIDManager(id_manager)
        # robot
        self._robot = robot_handler or Robot()
        self._robot_state = RobotStatus(self._robot)
        self._program = Program(self._robot)
        self.num_joints = self._robot.dof
        # if require_elbow_up:
        #     self._elbow_checker = ElbowChecker(self.num_joints, self._robot.robot_name)
        # # database
        self._db = DatabaseClient()
        # publishers
        self.joint_publish = self.create_publisher(JointState, 'joint_states', 10)
        self.pub_mjc_res = self.create_publisher(Bool, 'move_joint_to_cartesian/result', 10)
        self.pub_mjj_res = self.create_publisher(Bool, 'move_joint_to_joint/result', 10)
        self.pub_ml_res = self.create_publisher(Bool, 'move_linear/result', 10)
        self.pub_mlp_res = self.create_publisher(Bool, 'move_linear_via_points/result', 10)
        self.pub_mjv_res = self.create_publisher(Bool, 'move_joint_via_points/result', 10)
        self.pub_ctj = self.create_publisher(JointState, 'cartesian_to_joint_state', 10)
        self.pub_plan_mjj = self.create_publisher(String, 'plan_motion_joint_to_joint/result', 10)
        self.pub_plan_ml = self.create_publisher(String, 'plan_motion_linear/result', 10)
        self.pub_plan_mlp = self.create_publisher(String, 'plan_motion_linear_via_points/result', 10)
        self.pub_plan_mjv = self.create_publisher(String, 'plan_motion_joint_via_points/result', 10)
        self.pub_current_joint_state = self.create_publisher(JointState, 'current_joint_state', 10)
        self.pub_current_cartesian_pose = self.create_publisher(Pose, 'current_cartesian_pose', 10)
        self.pub_execute_if_successful = self.create_publisher(Bool, 'execute_if_successful/result', 10)
        self.pub_ik_solution = self.create_publisher(JointState, 'get_ik_solution/result', 10)
        self.pub_elbow_up_ik_solution = self.create_publisher(JointState, 'get_elbow_up_ik_solution/result', 10)
        self.pub_set_motion_till_force = self.create_publisher(Bool, 'set_motion_till_force/result', 10)
        self.pub_gripper_status = self.create_publisher(String, 'gripper_status', 10)
        # subscriptions
        self.create_subscription(Bool, 'get_current_joint_state', self.get_current_joint_state, 10)
        self.create_subscription(Bool, 'get_current_cartesian_pose', self.get_current_cartesian_pose, 10)
        self.create_subscription(Int32, 'execute_if_successful', self._execute_if_successful, 10)
        self.create_subscription(Pose, 'get_ik_solution', self.get_ik_solution, 10)
        self.create_subscription(Pose, 'get_elbow_up_ik_solution', self.get_elbow_up_ik_solution, 10)
        self.create_subscription(Float32MultiArray, 'set_motion_till_force', self.set_motion_till_force, 10)
        self.create_subscription(Pose, 'move_unified_pose', self.unified_pose_callback, 10)
        self.create_subscription(Bool, 'move_joint_to_cartesian', self.move_joint_to_cartesian, 10)
        self.create_subscription(JointState, 'move_joint_to_joint', self.move_joint_to_joint, 10)
        self.create_subscription(Bool, 'move_linear', self.move_linear, 10)
        self.create_subscription(PoseArray, 'move_linear_via_points', self.move_linear_via_points, 10)
        self.create_subscription(Int32, 'change_gripper', self._on_change_gripper_msg, 10)
        self.create_subscription(Int32MultiArray, 'execute_ids', self.execute, 10)
        self.create_subscription(JointState, 'plan_motion_joint_to_joint', self.plan_motion_joint_to_joint, 10)
        self.create_subscription(Pose, 'plan_motion_linear', self.plan_motion_linear, 10)
        self.create_subscription(PoseArray, 'plan_motion_linear_via_points', self.plan_motion_linear_via_points, 10)
        self.create_subscription(JointState, 'plan_motion_joint_via_points', self.plan_motion_joint_via_points, 10)
        self.create_subscription(Pose, 'cartesian_to_joint_state', self.cartesian_2_joint, 10)

    # Callback wrappers
    def _get_current_joint_state(self, msg):
        joints = self._get_current_joint_state()
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.position = joints
        self.pub_current_joint_state.publish(js)

    def _get_current_cartesian_pose(self, msg):
        pose = self.get_current_cartesian_tcp_pose()
        p = Pose()
        p.position.x, p.position.y, p.position.z, *_ = pose
        self.pub_current_cartesian_pose.publish(p)

    def _execute_if_successful(self, msg):
        res = self._execute_if_successful(msg.data)
        self.pub_execute_if_successful.publish(Bool(data=res))

    def _get_ik_solution(self, msg):
        sol = self.get_ik_solution([msg.position.x, msg.position.y, msg.position.z,
                                     msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                                     self.get_current_joint_state())
        js = JointState()
        js.position = sol
        self.pub_ik_solution.publish(js)

    def _get_elbow_up_ik_solution(self, msg):
        sol = self.get_elbow_up_ik_solution([msg.position.x, msg.position.y, msg.position.z,
                                              msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                                              self.get_current_joint_state())
        js = JointState()
        js.position = sol
        self.pub_elbow_up_ik_solution.publish(js)

    def move_joint_to_cartesian(self, msg):
        res = self.move_joint_to_cartesian([msg.position.x, msg.position.y, msg.position.z,
                                            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.pub_mjc_res.publish(Bool(data=res))

    def move_joint_to_joint(self, msg):
        res = self.move_joint_to_joint(msg.position)
        self.pub_mjj_res.publish(Bool(data=res))

    def move_linear(self, msg):
        res = self.move_linear([msg.position.x, msg.position.y, msg.position.z,
                                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.pub_ml_res.publish(Bool(data=res))

    def move_linear_via_points(self, msg):
        poses = [[p.position.x, p.position.y, p.position.z,
                  p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in msg.poses]
        res = self.move_linear_via_points(poses)
        self.pub_mlp_res.publish(Bool(data=res))

    def execute(self, msg):
        ids = list(msg.data)
        self.execute(ids, [True]*len(ids))

    def plan_motion_joint_to_joint(self, msg):
        ok, pid, last = self.plan_motion_joint_to_joint(msg.position)
        self.pub_plan_mjj.publish(String(data=f"{ok},{pid},{last}"))

    def plan_motion_linear(self, msg):
        ok, pid, last = self.plan_motion_linear([msg.position.x, msg.position.y, msg.position.z,
                                                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.pub_plan_ml.publish(String(data=f"{ok},{pid},{last}"))

    def plan_motion_linear_via_points(self, msg):
        poses = [[p.position.x, p.position.y, p.position.z,
                  p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w] for p in msg.poses]
        ok, pid, last = self.plan_motion_linear_via_points(poses)
        self.pub_plan_mlp.publish(String(data=f"{ok},{pid},{last}"))

    def plan_motion_joint_via_points(self, msg):
        traj = [pt.positions for pt in msg.points]
        ok, pid, last = self.plan_motion_joint_via_points(traj)
        self.pub_plan_mjv.publish(String(data=f"{ok},{pid},{last}"))

    def cartesian_2_joint(self, msg):
        joint_states = self.cartesian_2_joint([msg.position.x, msg.position.y, msg.position.z,
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        js = JointState()
        js.position = joint_states
        self.pub_ctj.publish(js)

    def unified_pose_callback(self, msg, _):
        # integrate database use
        code, ee = self._db.get_end_effector()
        if code.value < 0:
            self.get_logger().error('Failed to fetch EE')
        else:
            self.get_logger().info(f'Using EE: {ee.name}')

            
def main(args=None):
    rclpy.init(args=args)
    node = MairaKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# ### server.py

# import asyncio
# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionServer, CancelResponse, GoalResponse
# from geometry_msgs.msg import PoseStamped
# from maira_apps.action import MoveToPose
# from maira_kinematics import MairaKinematics

# class MairaActionServer(Node):
#     def __init__(self):
#         super().__init__('maira_action_server')
#         self._kin = MairaKinematics()
#         self._action_server = ActionServer(
#             self, MoveToPose, 'move_to_pose',
#             execute_callback=self.execute_callback,
#             goal_callback=self.goal_callback,
#             cancel_callback=self.cancel_callback)

#     def goal_callback(self, request):
#         return GoalResponse.ACCEPT

#     def cancel_callback(self, handle):
#         return CancelResponse.ACCEPT

#     async def execute_callback(self, handle):
#         feedback = MoveToPose.Feedback()
#         goal = handle.request.target_pose
#         ok, pid, _ = self._kin.plan_motion_linear([goal.pose.position.x,
#                                                   goal.pose.position.y,
#                                                   goal.pose.position.z,
#                                                   goal.pose.orientation.x,
#                                                   goal.pose.orientation.y,
#                                                   goal.pose.orientation.z,
#                                                   goal.pose.orientation.w])
#         if not ok:
#             handle.abort()
#             return MoveToPose.Result(success=False)
#         for i in range(1, 6):
#             if handle.is_cancel_requested:
#                 handle.canceled()
#                 return MoveToPose.Result(success=False)
#             feedback.progress = i*0.2
#             handle.publish_feedback(feedback)
#             await asyncio.sleep(0.2)
#         success = self._kin._execute_if_successful(pid)
#         result = MoveToPose.Result(success=success)
#         if success:
#             handle.succeed()
#         else:
#             handle.abort()
#         return result


# def main(args=None):
#     rclpy.init(args=args)
#     server = MairaActionServer()
#     rclpy.spin(server)
#     server.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()


### client.py

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient
# from geometry_msgs.msg import PoseStamped
# from maira_apps.action import MoveToPose

# class MairaActionClient(Node):
#     def __init__(self):
#         super().__init__('maira_action_client')
#         self._client = ActionClient(self, MoveToPose, 'move_to_pose')

#     def send_goal(self, pose: PoseStamped):
#         goal = MoveToPose.Goal(target_pose=pose)
#         self._client.wait_for_server()
#         self._client.send_goal_async(goal, feedback_callback=self.feedback_callback)

#     def feedback_callback(self, msg):
#         self.get_logger().info(f"Progress: {msg.feedback.progress*100:.1f}%")

#     def get_result_callback(self, future):
#         result = future.result().result
#         if result.success:
#             self.get_logger().info('Motion completed')
#         else:
#             self.get_logger().error('Motion failed')


# def main(args=None):
#     rclpy.init(args=args)
#     client = MairaActionClient()
#     pose = PoseStamped()
#     pose.header.frame_id = 'base_link'
#     pose.pose.position.x = 0.5
#     pose.pose.position.y = 0.0
#     pose.pose.position.z = 0.2
#     pose.pose.orientation.w = 1.0
#     client.send_goal(pose)
#     rclpy.spin(client)
#     client.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()