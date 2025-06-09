#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclpyDuration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from builtin_interfaces.msg import Duration
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
import tf2_ros
from geometry_msgs.msg import TransformStamped
try:
    from tf_transformations import euler_from_quaternion
except ImportError:
    def euler_from_quaternion(q):
        return tuple(R.from_quat(q).as_euler('xyz'))

# ——————————————————————————————————————————————————————————————————————————————
# Helpers & stubs
# ——————————————————————————————————————————————————————————————————————————————
def normalize_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap each element of angles to the range [-π, π)."""
    return (angles + np.pi) % (2.0 * np.pi) - np.pi

class calculation:
    Success = 1
    Failed  = 0

class IDManager:
    """Simply hands out incrementing integer IDs."""
    def __init__(self):
        self._id = 0
    def update_id(self) -> int:
        self._id += 1
        return self._id

class PlannerProgram:
    """Stub planner to record commands for plan_motion_* calls."""
    class _Cmd:
        Linear = object()
    def __init__(self):
        self.cmd = self._Cmd()
        self._last_joint = {}
    def set_command(self, cmd, cmd_id:int, current_joint_angles:list, reusable_id:int, **kwargs):
        self._last_joint[cmd_id] = current_joint_angles
    def get_last_joint_configuration(self, plan_id:int) -> list:
        return self._last_joint.get(plan_id, [])
    def get_plan_status(self, plan_id:int):
        return calculation.Success

try:
    from omniORB import CORBA
    from MairaCorba import Component
    _CORBA_AVAILABLE = True
except ImportError:
    _CORBA_AVAILABLE = False

# ——————————————————————————————————————————————————————————————————————————————
# URDF + IK handler
# ——————————————————————————————————————————————————————————————————————————————
class URDFChainHandler:
    def __init__(self, urdf_path:str, base_link:str="maira7M_root_link"):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")
        self.chain = Chain.from_urdf_file(urdf_path, base_elements=[base_link])
        # only actuated joints (links 2–8) are active
        self.actuated_indices = list(range(2,9))
        mask = [False]*len(self.chain.links)
        for i in self.actuated_indices:
            mask[i] = True
        self.chain.active_links_mask = mask

    def print_indices(self):
        print("[URDFChainHandler] Hard-coded joint indices:")
        for i in self.actuated_indices:
            print(f"  {i}: {self.chain.links[i].name}")

    def inverse_kinematics(self, target_position:np.ndarray, initial_joints:np.ndarray=None)->np.ndarray:
        if initial_joints is None:
            initial_joints = np.zeros(len(self.chain.links))
        sol = self.chain.inverse_kinematics(
            target_position,
            initial_position=initial_joints
        )
        return sol

# ——————————————————————————————————————————————————————————————————————————————
# Main node
# ——————————————————————————————————————————————————————————————————————————————
class MairaKinematics(Node):
    def __init__(self,
                 urdf_path:str,
                 program:PlannerProgram=None,
                 id_manager:IDManager=None,
                 robot_state_client=None):
        super().__init__('maira_kinematics')

        # IK handler
        self.ik = URDFChainHandler(urdf_path)
        self.program    = program    or PlannerProgram()
        self.id_manager = id_manager or IDManager()
        self.robot_state_client = robot_state_client

        # ROS2 interfaces
        self._current_state:JointState = None
        self.create_subscription(JointState,'/joint_states',self._joint_state_cb,10)

        # For TF2-based pose lookup
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer,self)

        # Single action client for all trajectory goals
        self.client = ActionClient(self,FollowJointTrajectory,
                                   '/joint_trajectory_position_controller/follow_joint_trajectory')

        # defaults
        self.speed_linear = 0.2   # m/s
        self.acc_linear   = 0.5   # m/s²
        self._last_err    = None
        self._waiting     = False

    # ——————————————————————————————————————————————————————————————————————————
    # State callbacks & getters
    # ——————————————————————————————————————————————————————————————————————————
    def _joint_state_cb(self,msg:JointState):
        self._current_state = msg

    def get_current_joint_state(self)->list[float]:
        """Prefer external interface, else /joint_states."""
        if self.robot_state_client:
            try:
                return self.robot_state_client.getRobotStatus("jointAngles")
            except Exception as e:
                self.get_logger().warn(f"robot_state error: {e}")
        if not self._current_state:
            self.get_logger().warn("No joint state available yet.")
            return []
        return list(self._current_state.position)

    def get_current_cartesian_pose(self)->list[float]:
        """
        Forward-kinematics via IK chain.
        Returns [x,y,z,roll,pitch,yaw].
        """
        js = self.get_current_joint_state()
        if not js:
            return []
        full = np.zeros(len(self.ik.chain.links))
        for i, idx in enumerate(self.ik.actuated_indices):
            full[idx] = js[i]
        T = self.ik.chain.forward_kinematics(full)
        xyz = T[:3,3]
        rpy = R.from_matrix(T[:3,:3]).as_euler('xyz',degrees=False)
        return [float(v) for v in (*xyz,*rpy)]

    def get_current_cartesian_pose_tf(self,
                                      source_frame='maira7M_root_link',
                                      target_frame='ee_link')->list[float]:
        """Lookup via TF2."""
        try:
            now = rclpy.time.Time()
            t:TransformStamped = self.tf_buffer.lookup_transform(
                source_frame,target_frame,now,
                timeout=RclpyDuration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(f"TF2 lookup failed: {e}")
            return []
        tr = t.transform.translation
        rt = t.transform.rotation
        rpy = euler_from_quaternion([rt.x,rt.y,rt.z,rt.w])
        return [tr.x,tr.y,tr.z,*rpy]

    # ——————————————————————————————————————————————————————————————————————————
    # Core send helper
    # ——————————————————————————————————————————————————————————————————————————
    def _send_goal(self,joint_names:list[str], points:list[JointTrajectoryPoint])->bool:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = joint_names
        goal.trajectory.points      = points
        goal.trajectory.header.stamp = self.get_clock().now().to_msg()

        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server unavailable")
            return False

        fut = self.client.send_goal_async(goal,
                                         feedback_callback=self._fb_cb)
        fut.add_done_callback(self._res_cb)
        return True

    def _build_point(self,positions:list[float], duration:float)->JointTrajectoryPoint:
        p = JointTrajectoryPoint()
        p.positions = positions
        sec = int(duration)
        p.time_from_start = Duration(sec=sec,
                                     nanosec=int((duration-sec)*1e9))
        return p

    def _fb_cb(self,feedback):
        # joint feedback
        names   = feedback.feedback.joint_names
        des     = feedback.feedback.desired.positions
        act     = feedback.feedback.actual.positions
        for nm,d,a in zip(names,des,act):
            self.get_logger().info(f"[fb] {nm}: des={d:.4f}, act={a:.4f}")
        # live TCP via IK
        pose6d = self.get_current_cartesian_pose()
        if pose6d:
            x,y,z,roll,pitch,yaw = pose6d
            self.get_logger().info(f" Live TCP: x={x:.3f},y={y:.3f},z={z:.3f}, roll={roll:.2f}")

    def _res_cb(self,future):
        res = future.result().result
        self.get_logger().info(f"Result code = {res.error_code}")
        # final TCP
        pose6d = self.get_current_cartesian_pose()
        if pose6d:
            x,y,z,roll,pitch,yaw=pose6d
            self.get_logger().info(f" Final TCP: x={x:.3f},y={y:.3f},z={z:.3f}")

    # ——————————————————————————————————————————————————————————————————————————
    # Trajectory validity
    # ——————————————————————————————————————————————————————————————————————————
    def is_trajectory_valid(self,traj:list[list[float]])->bool:
        try:
            n = len(self.ik.actuated_indices)
            if not isinstance(traj,list) or not traj:
                raise TypeError("Trajectory must be non-empty list.")
            for w in traj:
                if not isinstance(w,list) or len(w)!=n:
                    raise ValueError(f"Each waypoint needs {n} joints")
            return True
        except Exception as e:
            self.get_logger().error(f"Invalid trajectory: {e}")
            return False

    # ——————————————————————————————————————————————————————————————————————————
    # 1-point joint → joint
    # ——————————————————————————————————————————————————————————————————————————
    def move_joint_to_joint(self,target:list[float],duration:float=1.0)->bool:
        cur = self.get_current_joint_state()
        if not cur:
            return False
        if not self.is_trajectory_valid([target]):
            return False
        names = [f"joint{i+1}" for i in range(len(target))]
        pt = self._build_point(target,duration)
        return self._send_goal(names,[pt])

    # ——————————————————————————————————————————————————————————————————————————
    # Cartesian→joint (single-point)
    # ——————————————————————————————————————————————————————————————————————————
    def move_to_cartesian(self,xyz:list[float],duration:float=5.0)->bool:
        if not self._current_state:
            self.get_logger().warn("No joint state yet")
            return False
        # seed full chain
        seed = np.zeros(len(self.ik.chain.links))
        for i,idx in enumerate(self.ik.actuated_indices):
            seed[idx] = self._current_state.position[i]
        sol = self.ik.inverse_kinematics(np.array(xyz),seed)
        actu = normalize_to_pi(sol[self.ik.actuated_indices])
        names = [f"joint{i+1}" for i in range(len(actu))]
        return self._send_goal(names,[self._build_point(actu.tolist(),duration)])

    # ——————————————————————————————————————————————————————————————————————————
    # Elbow-up IK + send
    # ——————————————————————————————————————————————————————————————————————————
    def get_elbow_up_solution(self,xyz:list[float])->np.ndarray:
        if not self._current_state:
            raise RuntimeError("No joint state yet")
        base = np.zeros(len(self.ik.chain.links))
        for i,idx in enumerate(self.ik.actuated_indices):
            base[idx] = self._current_state.position[i]
        # fixed offset seeds
        presets=[(0,0),(0,np.pi),(np.pi,0),(np.pi,np.pi)]
        first=None
        for d6,d7 in presets:
            seed = base.copy()
            i6,i7 = self.ik.actuated_indices[4],self.ik.actuated_indices[5]
            seed[i6]+=((seed[i6]<0)*2-1)*d6
            seed[i7]+=((seed[i7]<0)*2-1)*d7
            try:
                sol=self.ik.inverse_kinematics(np.array(xyz),seed)
                if first is None: first=sol
            except: pass
        if first is not None: return first
        # random restarts
        for _ in range(10):
            seed=base.copy()
            seed[self.ik.actuated_indices]+=np.random.uniform(-np.pi,np.pi,7)
            try: return self.ik.inverse_kinematics(np.array(xyz),seed)
            except: pass
        raise ValueError("No IK solution")

    def move_cartesian_elbow_up(self,xyz:list[float],duration:float=5.0)->bool:
        sol = self.get_elbow_up_solution(xyz)
        actu = normalize_to_pi(sol[self.ik.actuated_indices])
        names=[f"joint{i+1}" for i in range(len(actu))]
        return self._send_goal(names,[self._build_point(actu.tolist(),duration)])

    # ——————————————————————————————————————————————————————————————————————————
    # Joint-via-points
    # ——————————————————————————————————————————————————————————————————————————
    def move_joint_via_points(self,
                               pts:list[list[float]],
                               times:list[float],
                               speed_scale:float=None)->bool:
        if not self.is_trajectory_valid(pts):
            return False
        names=[f"joint{i+1}" for i in range(len(pts[0]))]
        points=[]
        prev,prev_t=None,None
        for t_raw,p in zip(times,pts):
            t=(t_raw/speed_scale) if speed_scale else t_raw
            pt=JointTrajectoryPoint()
            pt.positions=p
            pt.time_from_start=Duration(sec=int(t),
                                        nanosec=int((t-int(t))*1e9))
            if prev is not None:
                dt=max(t-prev_t,1e-6)
                pt.velocities=[(c-p0)/dt for c,p0 in zip(p,prev)]
            else:
                pt.velocities=[0.0]*len(p)
            points.append(pt)
            prev,prev_t=p,t
        return self._send_goal(names,points)

    # ——————————————————————————————————————————————————————————————————————————
    # Linear-via-points
    # ——————————————————————————————————————————————————————————————————————————
    def move_linear_via_points(self,
                               pts:list[list[float]],
                               times:list[float],
                               speed_scale:float=None)->bool:
        return self.move_joint_via_points(pts,times,speed_scale)

    # ——————————————————————————————————————————————————————————————————————————
    # Plan-motion (stubs)
    # ——————————————————————————————————————————————————————————————————————————
    def is_id_successful(self,plan_id:int,timeout:float=10.0)->tuple[bool,bool]:
        start=time.time()
        status=self.program.get_plan_status(plan_id)
        while status not in (calculation.Success,calculation.Failed) and time.time()-start<timeout:
            time.sleep(0.01)
            status=self.program.get_plan_status(plan_id)
        return (status==calculation.Success,status==calculation.Success)

    def is_plan_successful(self,plan_id:int,timeout:float=10.0)->bool:
        ok,_ = self.is_id_successful(plan_id,timeout); return ok

    def clear_ids(self,ids:list[int])->bool:
        if not _CORBA_AVAILABLE:
            self.get_logger().warn("CORBA not available"); return False
        try:
            rts = Component(self.robot_state_client,"RTS")
            seq = CORBA.Any(CORBA.TypeCode("IDL:omg.org/CORBA/DoubleSeq:1.0"),ids)
            return (rts.callService("clearSplineId",[seq])==0)
        except Exception as e:
            self.get_logger().error(f"clear_ids error: {e}")
            return False

    def plan_motion_linear(self,
                           goal_pose:list[float],
                           start_cart=None,
                           start_js=None,
                           speed:float=None,
                           acc:float=None,
                           reusable:bool=False):
        if not (start_cart and start_js):
            start_cart=self.get_current_cartesian_pose()
            start_js  =self.get_current_joint_state()
        props={
            "target_pose":[start_cart,goal_pose],
            "speed":     speed or self.speed_linear,
            "acceleration": acc or self.acc_linear,
            "blending": False,
            "blend_radius":0.0
        }
        pid=self.id_manager.update_id()
        self.program.set_command(self.program.cmd.Linear,
                                 cmd_id=pid,
                                 current_joint_angles=start_js,
                                 reusable_id=int(reusable),
                                 **props)
        flags,_  = self.is_id_successful(pid),None
        last_j   = self.program.get_last_joint_configuration(pid) if all(flags) else []
        return flags,pid,last_j

    def plan_motion_joint_to_joint(self,
                                   goal_js:list[float],
                                   start_js:list[float]=None,
                                   speed:float=None,acc:float=None,
                                   reusable:bool=False):
        if start_js is None:
            start_js=self.get_current_joint_state()
        assert len(goal_js)==len(start_js)
        pid=self.id_manager.update_id()
        self.program.set_command(self.program.cmd.Linear,
                                 cmd_id=pid,
                                 current_joint_angles=start_js,
                                 reusable_id=int(reusable),
                                 target_pose=[start_js,goal_js],
                                 speed=speed or self.speed_linear,
                                 acceleration=acc or self.acc_linear,
                                 blending=False,blend_radius=0.0)
        flags,_=self.is_id_successful(pid),None
        last=self.program.get_last_joint_configuration(pid) if all(flags) else []
        return flags,pid,last

# ——————————————————————————————————————————————————————————————————————————————
# Example usage
# ——————————————————————————————————————————————————————————————————————————————
def main(args=None):
    rclpy.init(args=args)
    urdf = os.path.expanduser(
        '~/neura/sim_ws/src/neura_robot_description/maira7M/urdf/maira7M.urdf'
    )
    node = MairaKinematics(urdf)

    # wait for first joint state
    while rclpy.ok() and node._current_state is None:
        rclpy.spin_once(node,timeout_sec=0.1)

    node.get_logger().info("1) move_joint_to_joint")
    node.move_joint_to_joint([0.2,0.4,-0.1,0.0,0.1,-0.2,0.3],duration=2.0)

    # uncomment any of the following to test the rest:
    # node.move_to_cartesian([0.5,0.0,0.3],duration=4.0)
    # node.move_cartesian_elbow_up([0.3,-0.2,0.5],duration=5.0)
    # node.move_joint_via_points([[0,0.3,0.2,-0.1,0,0.1,0],[0.2,0.5,0.4,0.1,0.2,0.3,0.1]],[2.0,5.0])
    # node.move_linear_via_points([[0,0.3,0.2,-0.1,0,0.1,0],[0.2,0.5,0.4,0.1,0.2,0.3,0.1]],[2.0,5.0],speed_scale=1.2)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
