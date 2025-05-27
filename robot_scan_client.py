#!/usr/bin/env python3
"""Client to perform a scan of the robot's environment."""

from typing import Optional, Tuple

import actionlib
import rospy
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest

import neura_vision_tools_msgs.msg
import neura_vision_tools_msgs.srv

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes


class RobotScanClient(BaseAiClient):
    """Client to perform a scan of the robot's environment.

    Methods
    -------
    start_scanning
        Start scanning process with set of input.
    stop
        Stop scanning process and get scene as a mesh.
    resume
        Resume the scanning process after pausing.
    pause
        Pause the scanning process.
    hard_stop
        Force stop the scanning process even during robot motion.
    """

    def __init__(self):
        """Initialize client."""
        self._hard_stop_srv_proxy = rospy.ServiceProxy(
            "/neura_robot_scan/hard_stop_scanning", Trigger
        )
        self._stop_srv_proxy = rospy.ServiceProxy(
            "/neura_robot_scan/stop_scanning", Trigger
        )
        self._pause_srv_proxy = rospy.ServiceProxy(
            "/neura_robot_scan/pause_scanning", SetBool
        )
        self._srv_proxies = [
            self._hard_stop_srv_proxy,
            self._stop_srv_proxy,
            self._pause_srv_proxy,
        ]
        self._action_client = actionlib.SimpleActionClient(
            "/neura_robot_scan/scene_scanning_server",
            neura_vision_tools_msgs.msg.SceneReconstructionAction,
        )
        super(RobotScanClient, self).__init__(
            "/scene_reconstruction", self._srv_proxies, [self._action_client]
        )

    def start_scanning(
        self,
        workspace_name: str,
        file_name: str,
        cam_pose_type: int,
        scan_type: int,
        scene_id: Optional[int] = 0,
        do_texture_mapping: Optional[bool] = False,
        data_path: Optional[str] = "",
    ) -> ReturnCode:
        """Start scanning process with set of input.

        Parameters
        ----------
        workspace_name : str
            The name of available workspaces in the database
        file_name : str
            The file name of output mesh from scan process
        cam_pose_type : int
            type of method for estimating camera pose
            [0: use robot, 1: use marker, 2: use internal SLAM algorithm]
        scan_type : int
            type of scan
            [0: object scanning, 1: workspace scanning, 2: environment scanning]
        scene_id : int, optional
            id of the scene that robot scans, by default 0
        do_texture_mapping : bool, optional
            boolean option to run texture mapping process after scanning, by
            default False
        data_path : str, optional
            the directory to save scan output, by default ""

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        """
        self._log.debug("Sending goal")
        scan_client_info = neura_vision_tools_msgs.msg.ScanClientInfo()
        scan_client_info.camera_pose_type = cam_pose_type
        scan_client_info.scanning_type = scan_type
        scan_client_info.texture_mapping = do_texture_mapping
        scan_client_info.object_name = file_name
        scan_client_info.scene_id = scene_id
        scan_client_info.data_path = data_path
        goal = neura_vision_tools_msgs.msg.SceneReconstructionGoal()
        goal.scan_client_info = scan_client_info
        goal.workspace_name = workspace_name

        try:
            self._action_client.send_goal(goal, feedback_cb=self._feedback_cb)
        except rospy.ServiceException as ex:
            error_msg = f"Sending goal failed! \n {ex}"
            self._log.warning(error_msg)
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, error_msg)

        return ReturnCode(ReturnCodes.SUCCESS, "SUCCESS!")

    def _feedback_cb(self, feedback):
        """Callback that logs the feedback from the server.

        Parameters
        ----------
        feedback : __type__
            Feedback message from scan server

        """
        self._log.info(feedback)

    def stop(self) -> Tuple[ReturnCode, neura_vision_tools_msgs.msg.Mesh]:
        """Stop scanning process and get scene as a mesh.

        User should call the stop function when the motion of robot stops.
        After stop, the client publish the scene pointcloud to octomap server

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        neura_vision_tools_msgs.msg.Mesh
            The mesh of the scene with respect to robot link
        """
        trigger = TriggerRequest()
        # Now send the request through the connection
        self._log.debug("Send stop signal to action server")
        self._stop_srv_proxy.call(trigger)
        self._log.debug("Wait for result from action server")
        # Waits for the server to finish performing the action.
        self._action_client.wait_for_result()
        result = self._action_client.get_result()
        return result.rc, result.root_mesh

    def resume(self) -> ReturnCode:
        """Resume the scanning process after pausing.

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        """
        pause_scanning_request = SetBoolRequest()
        pause_scanning_request.data = False
        # Now send the request through the connection
        try:
            self._pause_srv_proxy.call(pause_scanning_request)
        except rospy.ServiceException as e:
            self._log.error(f"Resume scan failed! \n {e}")
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, repr(e))

        return ReturnCode(ReturnCodes.SUCCESS, "resume SUCCESS!")

    def pause(self) -> ReturnCode:
        """Pause the scanning process.

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        """
        pause_scanning_request = SetBoolRequest()
        pause_scanning_request.data = True
        # Now send the request through the connection
        try:
            self._pause_srv_proxy.call(pause_scanning_request)
        except rospy.ServiceException as ex:
            self._log.error(f"Pause scan failed! \n {ex}")
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, ex)
        return ReturnCode(ReturnCodes.SUCCESS, "pause SUCCESS!")

    def hard_stop(self) -> ReturnCode:
        """Force stop the scanning process even during robot motion.

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        """
        trigger = TriggerRequest()
        self._hard_stop_srv_proxy.call(trigger)
        try:
            self._hard_stop_srv_proxy.call(trigger)
        except rospy.ServiceException as ex:
            self._log.error("hard_stop scan failed: %s!", ex)
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, repr(ex))
        return ReturnCode(ReturnCodes.SUCCESS, "hard_stop SUCCESS!")
