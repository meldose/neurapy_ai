import rclpy # imported rclpy module
from rclpy.node import Node # imported node 
from rclpy.action import ActionClient # imported action client 
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint # imported TrajectoryPoint 
from builtin_interfaces.msg import Duration 
from sensor_msgs.msg import JointState #imported JointState
from std_msgs.msg import Bool # imported Bool
from geometry_msgs.msg import PoseArray, Pose # imported Pose

# created class for MOveJointTopJointClient
class MoveJointToJointClient(Node):

    def __init__(self):
        super().__init__('move_joint_to_joint_client')

        # Action client for joint trajectory
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )

        # Subscriber for cartesian waypoints
        self.create_subscription(
            PoseArray,
            '/cartesian_waypoints',  
            self.move_linear_via_points,
            10
        )

        # Publisher for success/failure
        self.pub_mlp_res = self.create_publisher(Bool, '/mlp_result', 10) 

        self.joint_names = [
            'maira7M_joint1','maira7M_joint2','maira7M_joint3',
            'maira7M_joint4','maira7M_joint5','maira7M_joint6',
            'maira7M_joint7',
        ]

        self._last_result_future = None
        self._last_result = None

# function for ik solver
    def _ik_solver(self, pose: Pose) -> list:
        """
        Stub IK solver. Replace with your actual IK implementation.
        Returns a list of joint positions matching self.joint_names.
        """
        self.get_logger().warn("IK solver not implemented; returning zero positions.")
        return [0.0] * len(self.joint_names)

# function for move linear through points
    def move_linear_via_points(self, msg: PoseArray):
        """
        Convert each Cartesian pose into a joint goal and execute sequentially.
        Publishes Bool on '/mlp_result' indicating overall success.
        """
        dt_between = 1.0  # seconds between waypoints

        for idx, pose in enumerate(msg.poses):
            # 1) Solve IK to get joint positions
            joint_state = JointState()
            joint_state.name = self.joint_names
            joint_state.position = self._ik_solver(pose)

            # 2) Send the joint goal
            success = self.send_joint_goal(joint_state, duration=dt_between)
            if not success:
                self.get_logger().error(f"Waypoint {idx} failed to send")
                self.pub_mlp_res.publish(Bool(data=False))
                return

            # 3) Wait for result before next waypoint
            rclpy.spin_until_future_complete(self, self._last_result_future)

            # Check execution result
            if self._last_result is None or self._last_result.error_code != 0:
                code = self._last_result.error_code if self._last_result else -1
                self.get_logger().error(f"Waypoint {idx} execution failed: error_code={code}")
                self.pub_mlp_res.publish(Bool(data=False))
                return

        self.pub_mlp_res.publish(Bool(data=True))

# function for sending the goal to the robot
    def send_joint_goal(self, joint_state: JointState, duration: float) -> bool:
        """
        Build and send a FollowJointTrajectory goal with a single waypoint.
        """
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_state.name

        # stamp the start time
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        # single point at target positions
        point = JointTrajectoryPoint()
        point.positions = joint_state.position

        # convert seconds to Duration
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal_msg.trajectory.points = [point]

        # send it off
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available!")
            return False

        self.get_logger().info(f"Sending joint goal (duration={duration}s)...")
        send_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_future.add_done_callback(self.goal_response_callback)
        return True
    
# function for response callback 
    def goal_response_callback(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().info("Goal rejected")
            return
        self.get_logger().info("Goal accepted")
        # store the future for synchronization
        self._last_result_future = handle.get_result_async()
        self._last_result_future.add_done_callback(self.get_result_callback)

# function for feedback callback 
    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f"Feedback: {feedback_msg.feedback}")

# fucntion for geting the result callback 
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: error_code={result.error_code}")
        # store last result for checking
        self._last_result = result

# defining the main function 
def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointClient()

    js = JointState()
    js.name = node.joint_names
    js.position = [0.5, 0.0, 0.5, -0.5, -0.2, -0.1, -0.2]
    success = node.send_joint_goal(js, duration=3.0)
    if not success:
        node.get_logger().error("Failed to send joint goal!")

    rclpy.spin(node)
    rclpy.shutdown()

# calling the main function
if __name__ == '__main__':
    main()
