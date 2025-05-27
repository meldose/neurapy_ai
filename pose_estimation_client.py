#!/usr/bin/env python3
"""Client to estimate object poses or train a new pose estimation model."""

from typing import List, Optional, Tuple, Sequence

import numpy as np
import numpy.typing as npt
import rospy
from cv_bridge import CvBridge

from neura_common_msgs.srv import GetMethod, GetModel, SelectMethod, SelectModel
from pose_estimation_ros_msgs.msg import Scene
from pose_estimation_ros_msgs.srv import (
    ClearViews,
    EstimatePoses,
    EstimatePosesImage,
    EstimatePosesMultiview,
    RegisterView,
)

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils import ros_conversions as conversions
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils.types import DetectedObject, ObjectWithPose


class PoseEstimationClient(BaseAiClient):
    """Client to estimate object poses or train a new pose estimation model.

    Object poses could be estimation using a trained pose estimation model or a
    new pose estimation model can be trained on a `neura-style` dataset.

    Methods
    -------
    set_model
        Load a new pose estimation model.
    get_model
        Get the currently loaded pose estimation model.
    set_method
        Load a new pose estimation method.
    get_method
        Get the currently used pose estimation method.
    get_poses
        Get pose estimates of chosen objects.
    multiview_pose_estimation_register_view
        Register the current camera view for multiview pose estimation.
    multiview_pose_estimation_clear_views
        Clear the stored camera views for multiview pose estimation.
    multiview_pose_estimation_get_poses
        Get pose estimates of chosen objects from the registered scene views.

    Notes
    -----
    - See `neurapy_ai.clients.DataGenerationClient` to generate your own
    `neura-style` dataset.

    """

    def __init__(
        self,
        model_name: Optional[str] = "",
        model_version: Optional[str] = "newest",
        refinement: Optional[bool] = True,
    ):
        """Initialize client.

        Parameters
        ----------
        model_name : str, optional
            Name of a pose estimation model to load at startup. The default
            is ''.
        model_version : str, optional
            Version of the pose estimation modle to laod. The default is
            'newest'.
        refinement : bool, optional
            If True, the initial pose estimates are further refined. The
            default is True.
        """
        self._pose_estimation_proxy = rospy.ServiceProxy(
            "/pose_estimation/estimate_poses", EstimatePoses
        )
        self._pose_estimation_image_proxy = rospy.ServiceProxy(
            "/pose_estimation/estimate_poses_image", EstimatePosesImage
        )

        self._multi_view_pose_estimation_proxy = rospy.ServiceProxy(
            "/pose_estimation/estimate_poses_multiview", EstimatePosesMultiview
        )
        self._register_view_proxy = rospy.ServiceProxy(
            "/pose_estimation/register_view", RegisterView
        )
        self._clear_views_proxy = rospy.ServiceProxy(
            "/pose_estimation/clear_views", ClearViews
        )
        self._select_model_proxy = rospy.ServiceProxy(
            "/pose_estimation/select_model", SelectModel
        )
        self._get_model_proxy = rospy.ServiceProxy(
            "/pose_estimation/get_model", GetModel
        )
        self._select_method_proxy = rospy.ServiceProxy(
            "/pose_estimation/select_method", SelectMethod
        )
        self._get_method_proxy = rospy.ServiceProxy(
            "/pose_estimation/get_method", GetMethod
        )

        service_proxies = [
            self._pose_estimation_proxy,
            self._multi_view_pose_estimation_proxy,
            self._register_view_proxy,
            self._clear_views_proxy,
            self._select_model_proxy,
            self._get_model_proxy,
            self._select_method_proxy,
            self._get_method_proxy,
        ]
        self.node_name = "pose_estimation"

        super(PoseEstimationClient, self).__init__(
            self.node_name, service_proxies, [], has_parameters=True
        )

        if model_name != "":
            self.set_model(model_name, model_version)

        self.set_parameter("icp_refinement", refinement)

        self._cam_frame = "camera_color_optical_frame"

        self.bridge = CvBridge()

    def set_model(
        self, model_name: str, model_version: Optional[str] = "newest"
    ) -> ReturnCode:
        """Load a new pose estimation model.

        Parameters
        ----------
        model_name : str
            Name of the pose estimation model to load
        model_version: str, optional
            Version of the pose estimation model that should be loadad, by
            default "newest".

        Returns
        -------
        ReturnCode
            Numerical return code with error message

        """
        try:
            result = self._select_model_proxy(model_name, model_version)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for setting the pose estimation model failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)
        return result.return_code

    def get_model(self) -> Tuple[str, str, ReturnCode]:
        """Get the currently loaded pose estimation model.

        Returns
        -------
        str
            Name of the loaded model
        str
            Version of the loaded model
        ReturnCode
            Numerical return code with error message

        """
        try:
            result = self._get_model_proxy()
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting the pose estimation model failed: %s",
                e,
            )
            return "", "", ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)
        return result.model_name, result.model_version, result.return_code

    def set_method(self, method: str) -> ReturnCode:
        """Load a new pose estimation method.

        This will clear all registered views for multiview pose estimation and
        reset the currently loaded model.

        Parameters
        ----------
        method : str
            Name of the pose estimation method to load

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        """
        try:
            result = self._select_method_proxy(method)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for setting the pose estimation method "
                f"failed: {e}"
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)
        return result.return_code

    def get_method(self) -> Tuple[str, ReturnCode]:
        """Get the current pose estimation method.

        Returns
        -------
        str
            Name of the current pose estimation method
        ReturnCode
            Numerical return code with error message

        """
        try:
            result = self._get_method_proxy()
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting the pose estimation method "
                f"failed: {e}"
            )
            return "", "", ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)
        return result.method, result.return_code

    def get_poses(
        self,
        class_names: Optional[Sequence[str]] = (),
        target_frame: Optional[str] = "",
    ) -> Tuple[ReturnCode, List[DetectedObject]]:
        """Get pose estimates of chosen objects.

        Get all object pose estimates or a filtered result based on given object
        names that match.

        Parameters
        ----------
        class_names : Sequence[str], optional
            A list of object class names, by default () (all classes).
        target_frame : str, optional
            Name of a coordinate frame in which the estimated poses should be
            returned, by default '' (the frame of the camera).

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        List[DetectedObject]
            A list of detected objects

        """
        detected_objects = []
        try:
            result = self._pose_estimation_proxy(class_names, target_frame)
        except rospy.ServiceException as e:
            self._log.error("Service call for pose estimation failed: %s", e)
            return (
                ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e),
                detected_objects,
            )

        return_code = ReturnCode()
        if result.return_code.value >= 0:
            detected_objects = self._msg_to_detected_objects(result.objects)
        else:
            self._log.error(
                "Calling pose estimation failed: " + result.return_code.message
            )
            return_code.value = result.return_code.value
            return_code.message = result.return_code.message
        return return_code, detected_objects

    def get_poses_from_image(
        self,
        color_image: npt.NDArray[np.uint8],
        depth_image: npt.NDArray[np.uint16],
        camera_intrinsics: npt.NDArray[np.float32],
        class_names: Optional[List[str]] = [],
        target_frame: Optional[str] = "",
        target_to_camera: npt.NDArray[np.float32] = np.eye(4, 4),
        timestamp: Optional[float] = None,
    ) -> Tuple[ReturnCode, List[DetectedObject]]:
        """Get pose estimates of chosen objects in the given scene.

        Get all object pose estimates or a filtered result based on given object
        names that match.

        Parameters
        ----------
        color_image : np.array (shape [height, width, 3], dtype uint8)
            The input color image in rgb channel order and 0-255 value range
        depth_image : np.array (shape [height, width, 1], dtype uint16)
            The input depth image with depth measurements in meter
        camera_intrinsics : np.array (shape [3, 3], dtype=float32)
            The camera intrinsics as 3 x 3 matrix
        class_names : List[str], optional
            A list of object class names, by default [] (all classes).
        target_frame : str, optional
            Name of a coordinate frame in which the estimated poses should be
            returned, by default '' (the frame of the camera).
        target_to_camera : np.array (shape [4, 4], dtype float32)
            Transformation between target frame and camera frame, by default
            np.eye(4,4).
        timestamp : float, optional
            Timestamp (in seconds) for the given scene, by default None
            (current time)

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        List[DetectedObject]
            A list of detected objects

        """

        if timestamp is None:
            timestamp = rospy.Time.now()
        else:
            timestamp = rospy.Time.from_sec(timestamp)
        scene = Scene()
        scene.color_image = self.bridge.cv2_to_imgmsg(color_image)
        scene.depth_image = self.bridge.cv2_to_imgmsg(depth_image)
        scene.camera_info = conversions.camera_intrinsics_2_camera_info_msg(
            color_image.shape, camera_intrinsics
        )
        if target_frame in ["", self._cam_frame]:
            target_frame = self._cam_frame
        else:
            if target_to_camera == np.eye(4, 4):
                self._log.warn(
                    "Target frame is not camera link but target_to_camera"
                    + " transformation is set to np.eye(4, 4)!"
                )

        target_to_cam = (
            conversions.transformation_matrix_2_geometry_msg_pose_stamped(
                target_to_camera, timestamp, target_frame
            )
        )
        scene.target_to_camera = target_to_cam
        scene.target_frame = target_frame

        detected_objects = []
        try:
            result = self._pose_estimation_image_proxy(class_names, scene)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for pose estimation for a given scene failed: %s",
                e,
            )
            return (
                ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e),
                detected_objects,
            )

        return_code = ReturnCode()
        if result.return_code.value < 0:
            self._log.error(
                "Running pose estimation for a given scene failed: "
                + result.return_code.message
            )
            return_code.value = result.return_code.value
            return_code.message = result.return_code.message
        else:
            for object_msg in result.objects:
                pose = conversions.geometry_msg_pose_2_pose(
                    object_msg.object_pose.pose
                )
                object_with_pose = ObjectWithPose(object_msg.class_name, pose)

                detected_object = DetectedObject(
                    object_with_pose,
                    object_msg.detection_score,
                    object_msg.pose_confidence,
                    object_msg.mesh_folder_path,
                    object_msg.segmentation_index,
                )
                detected_objects.append(detected_object)
            return_code.value = ReturnCodes.SUCCESS
            return_code.message = result.return_code.message

        return return_code, detected_objects

    def multiview_pose_estimation_register_view(
        self, class_names: Optional[Sequence[str]] = ()
    ) -> Tuple[ReturnCode, List[DetectedObject]]:
        """Register the current camera view for multiview pose estimation.

        Parameters
        ----------
        class_names : Sequence[str], optional
            A list of object class names, by default () (all classes).

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        List[DetectedObject]
            List of detected objects
        """
        detected_objects = []
        try:
            result = self._register_view_proxy(class_names)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for registering a view for multiview pose "
                + "estimation failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), []

        if result.return_code.value < 0:
            self._log.error(
                "Registering a view for multiview pose estimation failed: "
                + result.return_code.message
            )
            return (
                ReturnCode(
                    result.return_code.value, result.return_code.message
                ),
                [],
            )
        detected_objects = self._msg_to_detected_objects(result.objects)
        return ReturnCode(ReturnCodes.SUCCESS, ""), detected_objects

    def multiview_pose_estimation_clear_views(self) -> ReturnCode:
        """Clear the stored camera views for multiview pose estimation.

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        """
        try:
            result = self._clear_views_proxy()
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for clearing the stored views for multiview pose "
                + "estimation failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)

        if result.return_code.value < 0:
            self._log.error(
                "Clearing the stored views for multiview pose estimation "
                + "failed: "
                + result.return_code.message
            )
            return ReturnCode(
                result.return_code.value, result.return_code.message
            )

        return ReturnCode(ReturnCodes.SUCCESS, "")

    def multiview_pose_estimation_get_poses(
        self,
        class_names: Optional[Sequence[str]] = (),
        target_frame: Optional[str] = "",
    ) -> Tuple[ReturnCode, List[DetectedObject]]:
        """Get pose estimates of chosen objects from the registered scene views.

        Get pose estimates of all objects or a filtered result based on given
        object names that match from the registered views of the
        scene.

        Parameters
        ----------
        class_names : Sequence[str], optional
            A list of object class names, by default () (all classes).
        target_frame : str, optional
            Name of a coordinate frame in which the estimated poses should be
            returned, by default '' (the frame of the camera).

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        List[DetectedObject]
            List of detected objects
        """
        detected_objects = []
        try:
            result = self._multi_view_pose_estimation_proxy(
                class_names, target_frame
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for multi view pose estimation failed: %s", e
            )
            return (
                ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e),
                detected_objects,
            )

        return_code = ReturnCode()
        if result.return_code.value >= 0:
            detected_objects = self._msg_to_detected_objects(result.objects)
        else:
            self._log.error(
                "Running multiview pose estimation failed: "
                + result.return_code.message
            )
            return_code.value = result.return_code.value
            return_code.message = result.return_code.message
        return return_code, detected_objects

    @staticmethod
    def _msg_to_detected_objects(message):
        detected_objects = []
        for object_msg in message:
            pose = conversions.geometry_msg_pose_2_pose(
                object_msg.object_pose.pose
            )
            object_with_pose = ObjectWithPose(object_msg.class_name, pose)

            detected_object = DetectedObject(
                object_with_pose,
                object_msg.detection_score,
                object_msg.pose_confidence,
                object_msg.mesh_folder_path,
                object_msg.segmentation_index,
            )
            detected_objects.append(detected_object)
        return detected_objects
