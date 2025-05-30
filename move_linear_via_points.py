import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray


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
            '/joint_trajectory_position_controller/follow_joint_trajectory',
            self.move_linear_via_points,
            10
        )

        # Publisher for success/failure
        self.pub_mlp_res = self.create_publisher(Bool, 'mlp_result', 10)

        # your joint names
        self.joint_names = [
            'maira7M_joint1','maira7M_joint2','maira7M_joint3',
            'maira7M_joint4','maira7M_joint5','maira7M_joint6',
            'maira7M_joint7',
        ]

    def move_linear_via_points(self, msg: PoseArray):
        # ... existing cartesian path planning & send as before ...
        pass

    def send_joint_goal(self, joint_state: JointState, duration: float):
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

    def goal_response_callback(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().info("Goal rejected")
            return
        self.get_logger().info("Goal accepted")
        result_future = handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f"Feedback: {feedback_msg.feedback}")

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Result: error_code={result.error_code}")

def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointClient()

    # Example: send a single‐point joint goal
    js = JointState()
    js.name = node.joint_names
    js.position = [0.5, 0.0, -0.5, 0.5, 0.2, 0.1, 0.2]
    success = node.send_joint_goal(js, duration=3.0)
    if not success:
        node.get_logger().error("Failed to send joint goal!")

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
