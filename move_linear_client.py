import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState

class MoveJointToJointClient(Node):
    def __init__(self):
        super().__init__('move_joint_to_joint_client')
        self._client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_position_controller/follow_joint_trajectory'
        )
    def send_multi_waypoint_goal(self,
        joint_names: list[str],
        positions_list: list[list[float]],
        time_list: list[float],
        duration: float):

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.header.stamp = self.get_clock().now().to_msg()

        for pos, t in zip(positions_list, time_list):
            pt = JointTrajectoryPoint()

            pt.positions = [float(p) for p in pos]

            sec = int(t)
            nsec = int((t - sec) * 1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)

            goal_msg.trajectory.points.append(pt)

        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            rclpy.shutdown()
            return

        self.get_logger().info(f'Sending {len(goal_msg.trajectory.points)}-point trajectory…')
        send_goal_future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return
        self.get_logger().info('Goal accepted, waiting for result…')
        get_result = goal_handle.get_result_async()
        get_result.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.actual.positions}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: error_code={result.error_code}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointClient()

    joint_names = [
        'maira7M_joint1',
        'maira7M_joint2',
        'maira7M_joint3',
        'maira7M_joint4',
        'maira7M_joint5',
        'maira7M_joint6',
        'maira7M_joint7',
    ]

    positions_list = [
        [ 0.5, 0,    0,    0,    0,    0,    0],
        [-0.5, 0,    0,    0,    0,    0,    0],
        [ 0.5, 0,    0,    0,    0,    0,    0],
        [-0.5, 0,    0,    0,    0,    0,    0],
        [ 0.0, 0,    0,    0,    0,    0,    0],
    ]

    total_duration = 7.0
    num_pts = len(positions_list)
    time_list = [
        total_duration * (i + 1) / num_pts
        for i in range(num_pts)
    ] 

    node.get_logger().info('Sending multi-waypoint goal…')
    node.send_multi_waypoint_goal(
        joint_names,
        positions_list,
        time_list,
        total_duration
    )
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
