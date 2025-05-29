import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from your_package.action import MoveJointToJoint  # Import your generated action

class MoveJointToJointClient(Node):
    def __init__(self):
        super().__init__('move_joint_to_joint_client')
        self._client = ActionClient(self, MoveJointToJoint, 'move_joint_to_joint')

    def send_goal(self, joint_state: JointState):
        goal_msg = MoveJointToJoint.Goal()
        goal_msg.target_joint_state = joint_state

        self._client.wait_for_server()

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

        self.get_logger().info("Goal accepted")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'Feedback: {feedback_msg.feedback.status}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f' Result: success={result.success}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointClient()

    # Create a sample JointState goal
    joint_state = JointState()
    joint_state.name = ['joint_1', 'joint_2', 'joint_3']
    joint_state.position = [0.5, 1.0, -0.5]

    node.send_goal(joint_state)
    rclpy.spin(node)

if __name__ == '__main__':
    main()


#############################################



import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from your_package.action import MoveJointToJoint
from database_client_ros2 import DatabaseClientROS2


class MoveJointToJointActionClient(Node):
    def __init__(self):
        super().__init__('move_joint_to_joint_client')
        self._client = ActionClient(self, MoveJointToJoint, 'move_joint_to_joint')
        self._db_client = DatabaseClientROS2(self)
        self._db_client.wait_for_services()

    def send_goal_from_database(self, point_name: str):
        retcode, joints = self._db_client.get_joint_positions(point_name)
        if not retcode:
            self.get_logger().error(f"❌ DB Error: {retcode.message}")
            return

        joint_state = JointState()
        joint_state.name = [f'joint_{i+1}' for i in range(len(joints))]
        joint_state.position = joints

        goal_msg = MoveJointToJoint.Goal()
        goal_msg.target_joint_state = joint_state

        self._client.wait_for_server()
        self._send_goal_future = self._client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected')
            return
        self.get_logger().info('✅ Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(f'📡 Feedback: {feedback_msg.feedback.status}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'🎯 Result: {result.success}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MoveJointToJointActionClient()

    # Example database point name
    point_name = "home_position"
    node.send_goal_from_database(point_name)

    rclpy.spin(node)


if __name__ == '__main__':
    main()
