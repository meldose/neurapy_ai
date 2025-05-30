from rclpy.action import ActionClient
from my_robot_interfaces.action import MoveToCartesianPose
from geometry_msgs.msg import Pose
from std_msgs.msg import Bool

class MoveCartesianClient(Node):
    def __init__(self):
        super().__init__('move_cartesian_client')

        # 1) create the ActionClient for your Cartesian‐motion action
        self._cartesian_client = ActionClient(
            self,
            MoveToCartesianPose,
            '/move_to_cartesian_pose'
        )

        # 2) publisher for the Bool response
        self.pub_mjc_res = self.create_publisher(Bool, 'move_cartesian/result', 10)

        # 3) subscriber which will call our method on every Pose
        self.sub_pose = self.create_subscription(
            Pose,
            'move_cartesian/goal_pose',
            self.move_joint_to_cartesian,
            10
        )

    def move_joint_to_cartesian(self, msg: Pose):
        """Subscription callback — package the incoming Pose into an action goal."""
        self.get_logger().info(f"Received Cartesian goal: {msg}")
        self.send_cartesian_goal(msg)

    def send_cartesian_goal(self, pose: Pose, timeout: float = 5.0):
        """Builds and sends a MoveToCartesianPose goal."""
        goal_msg = MoveToCartesianPose.Goal()
        goal_msg.target = pose

        # Wait for the server to come up
        self._cartesian_client.wait_for_server()

        self.get_logger().info("Sending Cartesian goal…")
        send_goal_future = self._cartesian_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Cartesian goal rejected 😢")
            self.pub_mjc_res.publish(Bool(data=False))
            return

        self.get_logger().info("Cartesian goal accepted 👍")
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._get_result_callback)

    def _feedback_callback(self, feedback_msg):
        # optional: log or process partial feedback
        fb = feedback_msg.feedback
        self.get_logger().debug(f"Cartesian feedback: {fb}")

    def _get_result_callback(self, future):
        result = future.result().result
        success = bool(result.success)
        self.get_logger().info(f"Cartesian motion finished – success={success}")
        # publish the Bool result on your topic
        self.pub_mjc_res.publish(Bool(data=success))

def main(args=None):
    rclpy.init(args=args)
    node = MoveCartesianClient()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
