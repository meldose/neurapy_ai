import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from builtin_interfaces.msg import Duration

class MoveJointToJointClient(Node):
    def __init__(self):
        super().__init__('move_joint_to_joint_client')
        # 1) point at your active controller’s action server:
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )



    def send_goal(self, joint_state: JointState, duration: float = 7.0):
        """
        Send a FollowJointTrajectory goal to the action server.

        :param joint_state: JointState message containing names and target positions.
        :param duration: Time (in seconds) from trajectory start until waypoint is reached.
        """
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_state.name

        # Stamp the trajectory so the controller knows “when” to start it
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        # Create a single waypoint at the target positions
        point = JointTrajectoryPoint()
        point.positions = joint_state.position

        # Convert float seconds into Duration (sec + nanosec)
        sec = int(duration)
        nsec = int((duration - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)

        goal_msg.trajectory.points = [point]

        # Wait for server, send the goal, and hook up callbacks
        self._client.wait_for_server()
        self.get_logger().info(f'Sending goal with duration={duration}s...')
        self._send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: error_code={result.error_code}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointClient()
    joint_state = JointState()
    joint_state.name = ['maira7M_joint1', 'maira7M_joint2', 'maira7M_joint3', 'maira7M_joint4', 'maira7M_joint5', 'maira7M_joint6', 'maira7M_joint7']
    joint_state.position = [0.5, 0.0, -0.5, 0.5, 0.2, 0.1, 0.2]
    print('Sending Goal')

    #rclpy.spin(node)
    node.send_goal(joint_state,duration=7.0)

if __name__ == '__main__':
    main()
