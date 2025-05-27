#!/usr/bin/env python3
"""Client to generate grasps on any or known objects."""

from typing import List, Tuple

import rospy

from neurapy_ai.utils.sensor_data_acquisition import get_vision_data_async
from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.clients.database_client import DatabaseClient
from neurapy_ai.clients.audio_output_client import AudioOutputClient

from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils.types import Pick

from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from geometry_msgs.msg import PoseStamped, Point, TransformStamped
from neurapy_ai.utils.ros_conversions import pose_2_geometry_msg_pose
from instance_segmentation_ros_msgs.msg import (
    SegmentedInstance as SegmentedInstanceMsg,
)


class AutonomousPickClient(BaseAiClient):
    """Client to generate grasps on any or known objects.

    This does not require pre-recording of grasp poses and only optionally
    requires prior training an object segmentation model, which is when you are
    selective with the object that needs to be picked.

    Methods
    -------
    reset_GGC_parameters
        Reset GraspGeneratorClient parameters to defaults.
    start_detection
        Start grasp detection.
    get_picks
        Wait for the detection to complete and get a list of grasps.
    """

    def __init__(self):
        """Initialize client."""
        from neurapy_ai.experimental.clients.instance_segmentation_client import (
            InstanceSegmentationClient,
        )
        from neurapy_ai.experimental.clients.grasp_generator_client import (
            GraspGeneratorClient,
        )
        from neurapy_ai.experimental.clients.bin_detection_client import (
            BinDetectionClient,
        )

        self._picks = []
        self._return_code = ReturnCode()
        self._GGC = GraspGeneratorClient()
        self._ISC = InstanceSegmentationClient()

        self.reset_GGC_parameters()

        self._DBC = DatabaseClient()

        self._BDC = BinDetectionClient()
        self._BDC.set_parameter("bin_rim_cropping_factor", 0.3)
        self._BDC.set_parameter("pose_ws_refine", True)

        self._AOC = AudioOutputClient()

        self._node_name = "neura_grasp_pipeline"

        super(AutonomousPickClient, self).__init__(
            node_name=self._node_name,
            service_proxy=[],
            action_clients=[],
        )

        self._common_status_pub = rospy.Publisher(
            "/common_status", Int16, queue_size=1
        )

    def reset_GGC_parameters(self):
        """Reset GraspGeneratorClient parameters to defaults."""

        self._max_pick_attempts_per_capture = 1
        pick_params = {
            "enable_collision_checking_with_pointcloud": False,
            "enable_collision_checking_with_workspace": True,
            "enable_bin_localization": False,
            "workspace_offset_z": 0.0,
            "pre_grasp_distance": 0.1,
            "post_grasp_distance": 0.15,
            "max_num_grasps": 50,
            "thresh_rad": 0.4,
            "general_grasp_offset_x": 0.0,
            "general_grasp_offset_y": 0.0,
            "general_grasp_offset_z": 0.0,
            "collision_space_padding": 0.005,
            "approach_direction": 4,
            "ranking_method": 3,
            "random_pick_generation_method": 0,  # 0:gpd (larger objects), 1:grnet on plane (small objects)
            "end_effector_type": 0,  # 0: two fingered, 1: suction
        }
        self._GGC.set_parameters(pick_params)

    def start_detection(
        self,
        object_names: List[str],
        workspace_name: str,
        gripper_name: str,
        bin_name: str = "",
    ) -> ReturnCode:
        """Start grasp detection.

        Parameters
        ----------
        object_names : List[str]
            Objects that should be detected. If empty, grasps will be generated
            based on the point cloud in the region of interest
        workspace_name : str
            Name of the workspace that should be used for detection
        gripper_name : str
            Name of the gripper that should be used
        bin_name : str, optional
            Name of the bin that should be used. If empty, bin detection will
            not be used, by default ""

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

        self._picks = []

        # Signal to GUI to clear old grasp visuals from 3D scene
        self._common_status_pub.publish(Int16(-121))

        raw_point_cloud, raw_rgb_image, raw_depth_image, raw_camera_info = (
            get_vision_data_async(check_node_existence=False)
        )
        config = self._dyn_client.get_configuration()

        if config["enable_bin_localization"]:
            rc, bin_ws_db = self._DBC.get_workspace(bin_name)
            if not rc.value == ReturnCodes.SUCCESS:
                raise NameError(
                    f"No workspace named {bin_name} found in the database"
                )
            else:
                if bin_ws_db.mesh_model == "":
                    raise ValueError(
                        f"No mesh model assigned to named {bin_name}, cannot localize bin"
                    )

            rc, bin_pose_detected = self._BDC.detect_bin_pose_ros(
                bin_name=bin_ws_db.mesh_model,
                workspace_name=workspace_name,
                pointcloud_msg=raw_point_cloud,
                z_ws_bottom_offset=0,
                detection_method_type=1,  # default WITHOUT PREPROCESSING
            )
            if rc.value != ReturnCodes.SUCCESS:
                self._AOC.beep_negative()
                return ReturnCode(
                    ReturnCodes.SERVICE_CALL_FAILED, "Failed locating bin."
                )

            bin_pose_stamped = PoseStamped()
            bin_pose_stamped = bin_pose_detected.transformed_bin_pose_stamped

            bin_bbox_size = Point()
            # for inner dimension
            bin_bbox_size.x = (
                bin_pose_detected.bin_bbox_lbh.x - 0.04 * 0
            )  # 0.5645116769406753
            bin_bbox_size.y = (
                bin_pose_detected.bin_bbox_lbh.y - 0.03 * 0
            )  # 0.36972463222195123
            bin_bbox_size.z = (
                bin_pose_detected.bin_bbox_lbh.z - 0.03 * 0
            )  # 0.12

        elif not config["enable_bin_localization"] and bin_name != "":

            # TODO : question here, new bin detection works with different notation for bin workspace

            return_code, bin_ws = self._DBC.get_workspace(
                workspace_name=bin_name
            )

            if return_code.value != ReturnCodes.SUCCESS:
                self._AOC.beep_negative()
                raise KeyError(f"Workspace {bin_name} not found.")

            bin_pose_stamped = PoseStamped()
            bin_pose_stamped.pose = pose_2_geometry_msg_pose(bin_ws.pose)
            bin_pose_stamped.header.frame_id = bin_ws.frame

            bin_bbox_size = Point()
            bin_bbox_size.x = bin_ws.len_x
            bin_bbox_size.y = bin_ws.len_y
            bin_bbox_size.z = bin_ws.len_z

        else:

            bin_pose_stamped = None
            bin_bbox_size = None

            # run bin pose

        if object_names == []:
            pick_type = "general"
            segmentation_mask = (None,)
            segmented_instances = (None,)

        else:
            pick_type = "pose_agnostic"
            (
                segmented_instances,
                segmentation_mask,
            ) = self._execute_instance_segmentation(
                object_names=object_names, rgb_image_raw=raw_rgb_image
            )
            if len(segmented_instances) == 0:
                return ReturnCode(
                    ReturnCodes.SERVICE_CALL_FAILED,
                    "Failed segmenting object in scene.",
                )

        transform = TransformStamped()
        transform.header.frame_id = "root_link"

        self._GGC.generate_grasps(
            object_names=object_names,
            workspace_name=workspace_name,
            gripper_name=gripper_name,
            raw_point_cloud=raw_point_cloud,
            raw_rgb_image=raw_rgb_image,
            raw_depth_image=raw_depth_image,
            raw_camera_info=raw_camera_info,
            segmentation_mask=segmentation_mask,
            segmented_instances=segmented_instances,
            detected_poses=None,
            is_known_bin_pose=bin_pose_stamped is not None,
            bin_name=bin_name,
            bin_pose=bin_pose_stamped,
            bin_bbox=bin_bbox_size,
            check_reachability=False,
            pick_type=pick_type,
            tf_camera_to_root=transform,
        )
        # Sleep for reading data from camera
        rospy.sleep(0.5)
        return ReturnCode(ReturnCodes.SUCCESS, "Start detection")

    def get_picks(self) -> Tuple[ReturnCode, List[Pick]]:
        """Wait for the detection to complete and get a list of grasps.

        Please call `start_detection` before `get_picks` to start the detection
        thread.

        Returns
        -------
        ReturnCode
            Return code
        List[Pick]
            Detected grasps as `Pick` objects
        """

        self._return_code, self._picks = self._GGC.get_picks()
        return [self._return_code, self._picks]

    def _execute_instance_segmentation(
        self, object_names: list, rgb_image_raw: Image
    ) -> Tuple[List[SegmentedInstanceMsg], Image]:

        # ************** instance segmentation data **************
        rospy.loginfo(
            "[pick_app][instance segmentation data]: start instance segmentation"
        )

        (
            seg_return_code,
            segmented_instances,
            segmentation_mask,
        ) = self._ISC.get_segmented_instances_from_image_ros(
            rgb_image_raw,
            class_names=object_names,
        )
        if (
            seg_return_code.value < ReturnCodes.SUCCESS
            or len(segmented_instances) < 0
        ):
            self._AOC.beep_negative()
            return [], None

        rospy.loginfo(
            "[pick_app][instance segmentation data]: done instance segmentation"
        )

        return (
            segmented_instances,
            segmentation_mask,
        )
