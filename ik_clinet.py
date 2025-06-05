import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# Control‐related messages
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration


import PyKDL as kdl
from kdl_parser_py.urdf import treeFromParam

# ROS2 geometry conversions
from geometry_msgs.msg import Pose

class MoveJointToJointClient(Node):
    def __init__(self):
        super().__init__('move_joint_to_joint_client')

        #  Action client to the joint‐trajectory controller
        action_name = '/joint_trajectory_position_controller/follow_joint_trajectory'
        self._client = ActionClient(self, FollowJointTrajectory, action_name)

        #  Subscribe to /joint_states so we always have the “current joint positions”
        self._current_joint_state = None
        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

        success, kdl_tree = treeFromParam("robot_description")
        if not success:
            self.get_logger().error('Failed to parse /robot_description into a KDL tree')
            return


        self.base_link   = "base_link"   # ← replace with your robot’s actual base link name
        self.tip_link    = "ee_link"     # ← replace with your robot’s end‐effector link name

 
        self.kdl_chain = kdl_tree.getChain(self.base_link, self.tip_link)


        self.num_joints = self.kdl_chain.getNrOfJoints()


        self.lower_limits = kdl.JntArray(self.num_joints)
        self.upper_limits = kdl.JntArray(self.num_joints)

        self.chain_joint_names = []
        idx = 0
        for i in range(self.kdl_chain.getNrOfSegments()):
            seg = self.kdl_chain.getSegment(i)
            joint = seg.getJoint()
            if joint.getType() != kdl.Joint.None:
                jname = joint.getName()
                self.chain_joint_names.append(jname)

                urdf_node = self.get_parameter_or("robot_description", None).get_parameter_value().string_value

                from urdf_parser_py.urdf import URDF
                robot_urdf = URDF.from_parameter_server(self)

                joint_obj = robot_urdf.joint_map[jname]
                lower = joint_obj.limit.lower
                upper = joint_obj.limit.upper

                self.lower_limits[idx] = lower
                self.upper_limits[idx] = upper
                idx += 1


        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)


        vel_solver = kdl.ChainIkSolverVel_pinv(self.kdl_chain)
        self.ik_solver = kdl.ChainIkSolverPos_NR_JL(
            self.kdl_chain,
            self.lower_limits,
            self.upper_limits,
            self.fk_solver,
            vel_solver,
            maxiter=100,       # maximum Newton‐Raphson iterations
            eps=1e-6           # convergence tolerance
        )

        self.get_logger().info(f'IK solver ready for chain from "{self.base_link}" → "{self.tip_link}" '
                               f'with {self.num_joints} joints.')

    def _joint_state_callback(self, msg: JointState):
        """Store the most recent /joint_states message."""
        self._current_joint_state = msg

    def compute_ik(self, desired_pose: Pose, seed_positions: list[float]) -> list[float] | None:
        """
        Compute inverse kinematics for a given end-effector pose.

        :param desired_pose: geometry_msgs/Pose, containing .position (x,y,z) and .orientation (x,y,z,w).
        :param seed_positions: a Python list of length self.num_joints to seed the IK solver.
        :returns: Python list of joint angles [q1, q2, …] if IK succeeded, or None on failure.
        """
        #  Convert desired_pose → KDL.Frame:
        pos = desired_pose.position
        ori = desired_pose.orientation

        # KDL uses (w, x, y, z) ordering for quaternion:
        q_kdl = kdl.Rotation.Quaternion(ori.w, ori.x, ori.y, ori.z)
        trans_kdl = kdl.Vector(pos.x, pos.y, pos.z)
        target_frame = kdl.Frame(q_kdl, trans_kdl)

        #  Put seed_positions into a KDL.JntArray
        if len(seed_positions) != self.num_joints:
            self.get_logger().error(
                f"Seed length {len(seed_positions)} ≠ number of chain joints {self.num_joints}")
            return None

        seed_jnt = kdl.JntArray(self.num_joints)
        for i, val in enumerate(seed_positions):
            seed_jnt[i] = val

        #  Allocate an output JntArray
        result_jnt = kdl.JntArray(self.num_joints)

        #  Call the IK solver
        ik_result = self.ik_solver.CartToJnt(seed_jnt, target_frame, result_jnt)
        # return code == 0 → success
        if ik_result < 0:
            self.get_logger().warn(f"IK solver failed with code {ik_result}")
            return None

        #  Convert result_jnt → Python list[float]
        joint_angles = [result_jnt[i] for i in range(self.num_joints)]
        return joint_angles

    def send_goal(self, joint_names: list[str], joint_positions: list[float], duration: float):
        """
        Build and send a FollowJointTrajectory goal from a list of joint_names and joint_positions.
        This is almost identical to your original implementation, except we pass joint_names
        explicitly rather than copying from a JointState message.
        """
        goal_msg = FollowJointTrajectory.Goal()

        #  Copy joint names & header time
        goal_msg.trajectory.joint_names = joint_names[:]
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        #  Build exactly one JointTrajectoryPoint
        point = JointTrajectoryPoint()
        point.positions = joint_positions[:]

        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal_msg.trajectory.points = [point]

        #  Wait for the action server
        self.get_logger().info('Waiting for action server...')
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available after 5 seconds')
            return

        #  Send the goal
        self.get_logger().info(f'Sending goal (duration={duration}s)...')
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
        fb = feedback_msg.feedback
        self.get_logger().info(f"Received feedback: error_positions = {fb.error.positions}")

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result received: error_code = {result.error_code}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointClient()

    #  Wait for at least one /joint_states for up to 2 seconds
    t0 = node.get_clock().now()
    while rclpy.ok() and node._current_joint_state is None:
        if (node.get_clock().now().nanoseconds - t0.nanoseconds) > 2e9:
            node.get_logger().warn("No /joint_states received within 2 seconds. Continuing anyway.")
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    # If we do have a joint state, extract current joint positions in the same order as chain_joint_names:
    current_seed = [0.0] * node.num_joints
    if node._current_joint_state:
        # Build a map: joint_name → index in current_joint_state.position
        name2idx = {
            name: i for i, name in enumerate(node._current_joint_state.name)
        }
        for i, jname in enumerate(node.chain_joint_names):
            if jname in name2idx:
                current_seed[i] = node._current_joint_state.position[name2idx[jname]]
            else:
                node.get_logger().warn(f"Chain joint '{jname}' not in /joint_states, seeding 0.0.")

    desired_pose = Pose()
    desired_pose.position.x = 0.5
    desired_pose.position.y = 0.2
    desired_pose.position.z = 0.3

    desired_pose.orientation.w = 1.0
    desired_pose.orientation.x = 0.0
    desired_pose.orientation.y = 0.0
    desired_pose.orientation.z = 0.0

    #  Compute IK (seed it with current_seed)
    node.get_logger().info("Computing IK for desired end-effector pose...")
    ik_solution = node.compute_ik(desired_pose, current_seed)

    if ik_solution is None:
        node.get_logger().error("IK failed. Cannot send trajectory goal.")
    else:
        node.get_logger().info(f"IK succeeded. Joint solution: {ik_solution}")

        #  Send the resulting joint angles as a trajectory goal over 5 seconds
        node.send_goal(
            joint_names=node.chain_joint_names,
            joint_positions=ik_solution,
            duration=5.0
        )

        #  Spin until done
        rclpy.spin(node)

    # If IK failed or `rclpy.shutdown()` was called, we get here:
    node.get_logger().info("Exiting.")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
