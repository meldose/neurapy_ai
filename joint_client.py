
import rclpy #imported rclpy module
from rclpy.node import Node # imported Node module
from rclpy.action import ActionClient # imported Actionclient

from control_msgs.action import FollowJointTrajectory # imported FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # imported JointTrajectory and JointTrajectoryPoint
from sensor_msgs.msg import JointState # imported JointState
from builtin_interfaces.msg import Duration # imported Duration

# created class for MoveJointToJoinClient
class MoveJointToJointClient(Node):
    def __init__(self):
        super().__init__('move_joint_to_joint_client')

        #  Action client for FollowJointTrajectory
        action_name = '/joint_trajectory_position_controller/follow_joint_trajectory'
        self._client = ActionClient(self, FollowJointTrajectory, action_name)

        # Subscriber to /joint_states (published by joint_state_broadcaster)
        self._current_joint_state = None
        self._js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )

    def _joint_state_callback(self, msg: JointState):
        # Simply remember the latest incoming JointState message
        self._current_joint_state = msg

    def send_goal(self, joint_state: JointState, duration: float):
        goal_msg = FollowJointTrajectory.Goal()

        # Copy joint names from the provided JointState
        goal_msg.trajectory.joint_names = joint_state.name[:]

        # Stamp the trajectory header with the current time
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        # Build a single JointTrajectoryPoint
        point = JointTrajectoryPoint()
        point.positions = joint_state.position[:]
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal_msg.trajectory.points = [point]

        self.get_logger().info('Waiting for action server...')
        self._client.wait_for_server()

        self.get_logger().info(f'Sending goal (duration={duration}s)...')
        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

# created function for goal response callback
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal was rejected by the server.')
            return

        self.get_logger().info('Goal accepted → waiting for result...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

# created function for feedback callback
    def feedback_callback(self, feedback_msg):
        # Log any feedback from the action server
        self.get_logger().info(f'Feedback received: {feedback_msg.feedback}')

# created function for get result callback
    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result received: error_code = {result.error_code}')
        rclpy.shutdown()

# created main function
def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointClient()

    t0 = node.get_clock().now()
    while rclpy.ok() and node._current_joint_state is None:
        if (node.get_clock().now().nanoseconds - t0.nanoseconds) > 2e9:
            node.get_logger().warn("No /joint_states received within 2 seconds")
            break
        rclpy.spin_once(node, timeout_sec=0.1)

    if node._current_joint_state:
        node.get_logger().info(f"Current joint names: {node._current_joint_state.name}")
        node.get_logger().info(f"Current positions: {node._current_joint_state.position}")

    # --- Build a new JointState to send as a trajectory goal ---
    joint_state = JointState()
    joint_state.name = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7',
    ]
    joint_state.position = [-0.5, 0.0, 0.5, 0.8, 0.3, 0.4, 0.8]

    node.get_logger().info('Sending joint‐trajectory goal...')
    node.send_goal(joint_state, duration=8.0)

    rclpy.spin(node)

# calling main function
if __name__ == '__main__':
    main()
