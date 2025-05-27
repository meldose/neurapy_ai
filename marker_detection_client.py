#!/usr/bin/env python3
"""Client to detect and interpret (ArUco/ChArUco/Chessboard) markers."""

from typing import Tuple, List, Union
from time import time
import rospy
from neura_marker_detection_msgs.msg import DetectedTags
from neura_marker_detection_msgs.srv import (
    UpdateTargetParams,
    UpdateTargetParamsRequest,
)
from std_srvs.srv import SetBool, SetBoolRequest

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils.ros_conversions import geometry_msg_pose_2_pose
from neurapy_ai.utils.types import Marker, Pose


class MarkerDetectionClient(BaseAiClient):
    """Client to detect and interpret (ArUco/ChArUco/Chessboard) marker.

    Methods
    -------
    get_detected_marker
        Detect marker and return its pose.
    get_detected_markers
        Detect marker and return its pose.
    get_calibration_grid
        Gets the pose of the Charuco.
    activate_detection
        Activate marker detection stream.
    deactivate_detection
        Deactivate marker detection stream.
    """

    def __init__(self):
        """Initialize client."""
        self._markers = []
        self._active = False
        self._node_name = "neura_marker_detection"
        self._detection_timeout = 5  # 5 sec
        self._service_name = "/neura_marker_detection/update_target_params"
        self._detect_marker_proxy = rospy.ServiceProxy(
            self._service_name, UpdateTargetParams
        )

        service2_name = "/neura_marker_detection/start_detection"
        self._start_detection_proxy = rospy.ServiceProxy(service2_name, SetBool)

        sub_topic = "/neura_marker_detection/detected_markers"
        self._marker_sub = rospy.Subscriber(
            sub_topic, DetectedTags, self._marker_detection
        )

        super(MarkerDetectionClient, self).__init__(
            node_name=self._node_name,
            service_proxy=[
                self._detect_marker_proxy,
                self._start_detection_proxy,
            ],
            action_clients=[],
            has_parameters=False,
        )

    def _marker_detection(self, marker_msg: DetectedTags):
        """
        Marker Detection Service Callback

        Parameters
        ----------
        marker_msg : DetectedTags
            Detected Marker
        """
        if self._active:
            for pose_idx, marker_pose in enumerate(marker_msg.poses):
                neura_pose = geometry_msg_pose_2_pose(marker_pose.pose)
                marker_id = marker_msg.marker_ids[pose_idx]
                self._markers.append(Marker(marker_id, neura_pose))
            self._active = False

    def get_detected_marker(
        self,
        marker_type: int,
        vertical_squares: int,
        horizontal_squares: int,
        square_length: float,
        marker_dictionary: str,
        marker_length: float,
        timeout: int = 5,
    ) -> Tuple[ReturnCode, Union[None, List[Marker]]]:
        """Detect markers and return correspondent poses.

        Parameters
        ----------
        marker_type : int
            0: Aruco, 1: Charuco, 2: Chessboard
        vertical_squares : int
            Number of squares in the vertical direction.
        horizontal_squares : int
            Number of squares in the horizontal direction.
        square_length : float
            Length of a single square in meters.
        marker_dictionary : str
            Dictionary of markers to use. Supported options:
            "DICT_4X4_250", "DICT_5X5_250", "DICT_6X6_250", "DICT_7X7_250",
            "DICT_ARUCO_ORIGINAL". Default: "DICT_5X5_250".
        marker_length : float
            Length of the marker square including white borders. Default: 0.03 meters.
        timeout : int, optional
            Timeout in seconds to stop the marker detection if no marker is found. Default: 5.

        Returns
        -------
        Tuple[ReturnCode, Union[None, List[Marker]]]
            Return code and a list of detected markers, or None if no marker is found.
        """
        # RESET
        self._markers = []
        self._detection_timeout = timeout

        try:
            rospy.wait_for_service(self._service_name, 2.0)
        except (rospy.ServiceException, rospy.ROSException) as e:
            self._log.error("Service connection failed: %s", e)
            return (
                ReturnCode(
                    ReturnCodes.SERVICE_NOT_AVAILABLE,
                    ("Service connection failed: %s", e),
                ),
                None,
            )

        req = UpdateTargetParamsRequest()
        # translate from 'pixel numbers' to openCV marker id
        if marker_dictionary not in [
            "DICT_4X4_250",
            "DICT_5X5_250",
            "DICT_6X6_250",
            "DICT_7X7_250",
            "DICT_ARUCO_ORIGINAL",
        ]:
            self._log.error(f"Wrong dictionary type: {marker_dictionary}")
            return (
                ReturnCode(
                    ReturnCodes.INVALID_ARGUMENT,
                    "Supported Aruco markers dictionaries are 4, 5, 6, 7 "
                    "(numbers of single 'pixels' that builds the marker)",
                ),
                None,
            )

        req.dictionary_id = marker_dictionary
        req.optical_frame = "camera_color_optical_frame"
        req.marker_detector_type = marker_type  #
        req.squares_in_x = (
            vertical_squares  # number of elements in x(vertical) direction
        )
        req.squares_in_y = (
            horizontal_squares  # number of elements in y(horizontal) direction
        )
        req.square_size_m = (
            square_length  # square marker size including black part
        )
        req.marker_size_m = marker_length  # square with white border

        # Activate Marker Detection Stream
        activ_ret = self.activate_detection()
        # Wait untill stream is activated
        rospy.sleep(0.1)
        # Activate callback
        self._active = True

        if activ_ret.value >= 0:
            response = self._detect_marker_proxy(req)
            if not response.success:
                rospy.logerr(response.message)
                self.deactivate_detection()
                return (
                    ReturnCode(
                        response.success,
                        response.message,
                    ),
                    None,
                )
        else:
            self.deactivate_detection()
            return activ_ret, None
        t0 = time()
        while self._active:
            if int(time() - t0) > self._detection_timeout:
                return (
                    ReturnCode(
                        ReturnCodes.SERVICE_CALL_FAILED,
                        f"No marker detected after {self._detection_timeout}"
                        " seconds",
                    ),
                    None,
                )
            rospy.sleep(0.1)
        self.deactivate_detection()
        return ReturnCode(), self._markers

    def get_calibration_grid(self) -> Tuple[ReturnCode, Pose]:
        """Get pose of a big (14x9) calibration board.
        Note: this function is added for backwards compatiblity

        Returns
        -------
        ReturnCode
            Return code with message
        Pose
            Detected calibration board pose
        """
        marker_dict = {
            0: "DICT_4X4_250",
            1: "DICT_5X5_250",
            2: "DICT_6X6_250",
            3: "DICT_7X7_250",
            4: "DICT_ARUCO_ORIGINAL",
        }
        rc, marker = self.get_detected_marker(
            marker_type=1,
            vertical_squares=14,
            horizontal_squares=9,
            square_length=0.04,
            marker_dictionary=marker_dict[1],
            marker_length=0.03,
            timeout=5,
        )
        return rc, marker.pose

    def get_detected_markers(
        self,
        marker_dictionary: int,
        marker_length: float,
        marker_type: int = 1,
        vertical_squares: int = 9,
        horizontal_squares: int = 14,
        square_length: float = 0.04,
        timeout: int = 5,
    ) -> Tuple[ReturnCode, List[Marker]]:
        """Detect marker and return its pose.
        Note: this function is added for backwards compatibility.

        Parameters
        ----------
        marker_dictionary : int
            Currently only these markers are supported:
            0: "DICT_4X4_250", 1: "DICT_5X5_250", 2: "DICT_6X6_250",
            3: "DICT_7X7_250", 4: "DICT_ARUCO_ORIGINAL".
        marker_length : float
            Square length with white borders.
        marker_type : int, optional (default=1)
            0: Aruco, 1: Charuco, 2: Chessboard.
        vertical_squares : int, optional (default=9)
            Number of squares in the vertical direction.
        horizontal_squares : int, optional (default=14)
            Number of squares in the horizontal direction.
        square_length : float, optional (default=0.04)
            Length of the marker square.
        timeout : int, optional (default=5)
            Timeout to stop the marker detection if no marker was detected.

        Returns
        -------
        ReturnCode
            Return code with message.
        List[Marker]
            List of detected markers.
        """
        marker_dict = {
            0: "DICT_4X4_250",
            1: "DICT_5X5_250",
            2: "DICT_6X6_250",
            3: "DICT_7X7_250",
            4: "DICT_ARUCO_ORIGINAL",
        }
        rc, markers = self.get_detected_marker(
            marker_type=marker_type,
            vertical_squares=vertical_squares,
            horizontal_squares=horizontal_squares,
            square_length=square_length,
            marker_dictionary=marker_dict[marker_dictionary],
            marker_length=marker_length,
            timeout=timeout,
        )

        return rc, markers

    def activate_detection(self) -> ReturnCode:
        """Activate marker detection stream."""
        try:
            active_req = SetBoolRequest()
            active_req.data = True
            active_res = self._start_detection_proxy(active_req)
            if active_res:
                return ReturnCode(ReturnCodes.SUCCESS, "")
            else:
                return ReturnCode(
                    ReturnCodes.SERVICE_CALL_RETURN_ERROR,
                    "Marker Detection Activation Stream was not Possible!",
                )

        except rospy.ServiceException as e:
            self._log.error("Service call failed: %s", e)
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)

    def deactivate_detection(self) -> ReturnCode:
        """Deactivate marker detection stream."""
        deactive_req = SetBoolRequest()
        deactive_req.data = False
        deactive_res = self._start_detection_proxy(deactive_req)
        if deactive_res:
            return ReturnCode(ReturnCodes.SUCCESS, "")
        else:
            return ReturnCode(
                ReturnCodes.SERVICE_CALL_RETURN_ERROR,
                "Marker Detection Deactivation Stream was not Possible!",
            )
