#!/usr/bin/env python3
"""Client to generate grasps for objects that a human operator points to."""

from typing import List, Optional, Tuple

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus

import neurapy_ai.utils.ros_conversions as rc
from neurapy_ai.utils.sensor_data_acquisition import get_vision_data_async
from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils.types import (
    ApproachSequence,
    Gripper,
    JointState,
    ObjectWithPose,
    Pick,
    TCPPose,
)
from neura_grasp_planning_msgs.msg import (
    GenerateGraspsAction,
    GenerateGraspsGoal,
)


class PointingPickClient(BaseAiClient):
    """Client to generate grasps for objects that a human operator points to.

    The operator would be prompted to point their forefinger to the object
    within the chosen workspace for which grasps are to be generated.

    Methods
    -------
    start_detection
        Start grasp detection.
    get_picks
        Wait for the detection to complete and get a list of grasps.
    stop
        Stop detection.
    """

    def __init__(self):
        """Initialize client."""
        self._picks = []
        self._return_code = ReturnCode()
        self._node_name = "neura_grasp_pipeline"
        self._pointing_pick_client = actionlib.SimpleActionClient(
            "/neura_grasp_planner/generate_grasps",
            GenerateGraspsAction,
        )
        super(PointingPickClient, self).__init__(
            node_name=self._node_name,
            service_proxy=[],
            action_clients=[self._pointing_pick_client],
        )
        self.gripper_name = None
        self.object_name = None

    def start_detection(
        self,
        workspace_name: str,
        gripper_name: str,
        object_name: Optional[str] = "",
    ) -> ReturnCode:
        """Start grasp detection.

        Parameters
        ----------
        workspace_name : str
            Name of the workspace that should be used for detection.
        gripper_name : str
            Name of the gripper that should be used.
        object_name : str
            The name of the object that is selected. If empty, a random object
            will be assumed, by default "".

        Returns
        -------
        ReturnCode
            Return code

        Raises
        ------
        ValueError
            If `gripper_name` is empty
        """
        if gripper_name == "":
            msg = "Gripper id cannot be empty!"
            raise ValueError(msg)

        # Save as global value

        raw_point_cloud, raw_rgb_image, raw_depth_image, raw_camera_info = (
            get_vision_data_async(check_node_existence=False)
        )

        config = self._dyn_client.get_configuration()

        self.gripper_name = gripper_name
        self.object_name = object_name

        goal = GenerateGraspsGoal()

        goal.workspace_name = workspace_name
        goal.end_effector_name = gripper_name
        goal.type = GenerateGraspsGoal.POINTING
        goal.raw_point_cloud = raw_point_cloud
        goal.raw_rgb_image = raw_rgb_image
        goal.raw_depth_image = raw_depth_image
        goal.raw_camera_info = raw_camera_info
        goal.check_reachability = False

        self._picks = []
        try:
            self._pointing_pick_client.send_goal(goal)
            # Sleep for reading data from camera
            rospy.sleep(0.2)
        except rospy.ServiceException as e:
            self._log.warn("Action call failed: %s" % e)
            return ReturnCode(
                ReturnCodes.SERVICE_CALL_FAILED, "Service call failed!"
            )
        return ReturnCode(ReturnCodes.SUCCESS, "Start detection")

    def get_picks(self) -> Tuple[ReturnCode, List[Pick]]:
        """Wait for the detection to complete and get a list of grasps.

        Please call `start_detection` before `get_picks` to start the detection
        process.

        Returns
        -------
        ReturnCode
            Return code
        List[Pick]
            Detected grasps as `Pick` objects
        """
        if self._pointing_pick_client.get_state() is GoalStatus.LOST:
            self._return_code = ReturnCode(
                ReturnCodes.FUNCTION_NOT_INITIALIZED,
                "No detection was triggered",
            )
            return [self._return_code, self._picks]

        # Wait until the thread joint
        self._pointing_pick_client.wait_for_result(rospy.Duration(30))
        res = self._pointing_pick_client.get_result()
        if res is not None:
            self._log.debug("Action call returned")
            self._log.debug(f"Get {len(self._picks)} picks.")
        else:
            self._pointing_pick_client.cancel_all_goals()
            self._return_code = ReturnCode(
                ReturnCodes.SERVICE_CALL_FAILED,
                "Service did not return valid result (e.g. time exceed.)",
            )
            return [self._return_code, []]

        for instance_pick_pose in res.instance_pick_poses:
            for pick_pose_msg in instance_pick_pose.picks:
                pre_pick_pose = TCPPose(
                    rc.geometry_msg_pose_2_pose(
                        pick_pose_msg.pre_grasp_pose.pose
                    ),
                    JointState(pick_pose_msg.pre_grasp_js),
                )
                pick_pose = TCPPose(
                    rc.geometry_msg_pose_2_pose(pick_pose_msg.grasp_pose.pose),
                    JointState(pick_pose_msg.grasp_js),
                )
                post_pick_pose = TCPPose(
                    rc.geometry_msg_pose_2_pose(
                        pick_pose_msg.post_grasp_pose.pose
                    ),
                    JointState(pick_pose_msg.post_grasp_js),
                )
                approach_sequence = ApproachSequence(
                    pre_pick_pose, pick_pose, post_pick_pose
                )
                gripper = Gripper(
                    self.gripper_name, 0.0, pick_pose_msg.hand_opening + 0.02
                )
                object_with_pose = ObjectWithPose(
                    pick_pose_msg.object_name,
                    rc.geometry_msg_pose_2_pose(pick_pose_msg.object_pose),
                )
                pick = Pick(
                    approach_sequence=approach_sequence,
                    quality=pick_pose_msg.grasp_quality,
                    gripper=gripper,
                    object_with_pose=object_with_pose,
                    grasp_id=pick_pose_msg.grasp_idx,
                )
                self._picks.append(pick)
        self._return_code = ReturnCode(
            ReturnCodes.SUCCESS, "Call server succeed!"
        )
        # Reset thread to None
        return [self._return_code, self._picks]

    def stop(self) -> None:
        """Stop detection."""
        self._pointing_pick_client.cancel_all_goals()
