#!/usr/bin/env python3
"""Client to detect the location that a human operator points to."""

from typing import Tuple

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus

from neura_pointing_point_detection_msgs.msg import (
    DetectPointingPointAction,
    DetectPointingPointGoal,
)

import neurapy_ai.utils.ros_conversions as rc
from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils.types import Pose


class PointingPointDetectionClient(BaseAiClient):
    """Client to detect the location that a human operator points to.

    The operator is prompted to point with their forefinger to a
    desired location on a workspace surface. Their hand should be in the camera
    view and the projected pointed location on the surface is detected.

    Methods
    -------
    start_detection
        Start detecting the pointed location.
    get_point
        Wait for the detection to complete and get the desired location.
    stop
        Stop detection.
    """

    def __init__(self):
        """Initialize client."""
        self._node_name = "neura_pointing_point_detection"
        self._detect_pointing_point_client = actionlib.SimpleActionClient(
            "/" + self._node_name + "/" + "detect_pointing_point",
            DetectPointingPointAction,
        )
        super(PointingPointDetectionClient, self).__init__(
            node_name=self._node_name,
            service_proxy=[],
            action_clients=[
                self._detect_pointing_point_client,
            ],
        )

    def start_detection(self, workspace_name: str) -> ReturnCode:
        """
        Start detecting the pointed location.

        Th ehuman operator will be prompted to point their forefinger
        towards the desired location on the chosen workspace surface. Their hand
        should be within the camera view.

        Parameters
        ----------
        workspace_name : str
            The name of a registered workspace

        Returns
        -------
        ReturnCode
            Return code
        """
        goal = DetectPointingPointGoal()
        goal.workspace_name = workspace_name
        self._pointing_point = None
        try:
            self._detect_pointing_point_client.send_goal(goal)
        except rospy.ServiceException as e:
            self._log.warn("Action call failed: %s" % e)
            return ReturnCode(
                ReturnCodes.SERVICE_CALL_FAILED, "Service call failed!"
            )
        return ReturnCode(ReturnCodes.SUCCESS, "")

    def get_point(self) -> Tuple[ReturnCode, Pose]:
        """Wait for the detection to complete and get the desired location.

        Please call `start_detection` before `get_point` to start the detection
        thread.

        Returns
        -------
        ReturnCode
            Return code
        Pose
            The detected point relative to the robot
        """
        if self._detect_pointing_point_client.get_state() is GoalStatus.LOST:
            return_code = ReturnCode(
                ReturnCodes.FUNCTION_NOT_INITIALIZED,
                "No detection was triggered",
            )
            return (return_code, None)

        # Wait until the thread joint
        self._detect_pointing_point_client.wait_for_result(rospy.Duration(30))
        res = self._detect_pointing_point_client.get_result()
        state = self._detect_pointing_point_client.get_state()
        if res is not None and state == 3:
            self._log.debug("Action call returned")
        else:
            self._detect_pointing_point_client.cancel_all_goals()
            return_code = ReturnCode(
                ReturnCodes.SERVICE_CALL_FAILED,
                "Service did not return valid result (e.g. time exceed.)",
            )
            return [return_code, None]

        # Reset thread to None
        return_code = ReturnCode(ReturnCodes.SUCCESS, "Call server succeed!")
        return (return_code, rc.geometry_msg_pose_2_pose(res.pointing_point))

    def stop(self) -> None:
        """Stop detection."""
        self._detect_pointing_point_client.cancel_all_goals()
