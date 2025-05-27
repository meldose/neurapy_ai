#!/usr/bin/env python3
from typing import List, Tuple, Any, Optional
import genpy
import neura_vision_tools_msgs.msg
from neurapy_ai.utils.types import ObjectWithPose
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.clients.base_ai_client import BaseAiClient
import rospy
from sensor_msgs.msg import PointCloud2
from neura_bin_detection_msgs.msg import MethodType
from neura_bin_detection_msgs.srv import (
    DetectBinPose,
    DetectBinPoseRequest,
    GetDetectedBinPose,
    GetDetectedBinPoseRequest,
)
import neurapy_ai.utils.ros_conversions as rc


class BinDetectionClient(BaseAiClient):
    def __init__(self) -> None:
        """
        Initializes BinDetection client.

        Raises:
            ConnectionError: If the bin detection services are not
            available
        """
        self._return_code = ReturnCode()
        self._node_name = "neura_bin_detection"

        (
            self._detect_bin_pose_client,
            self._get_bin_pose_client,
        ) = self._initialize_clients(
            [
                "detect_bin_pose",
                "get_detected_bin_pose",
            ],
            [DetectBinPose, GetDetectedBinPose],
            self._node_name,
        )

        super(BinDetectionClient, self).__init__(
            node_name=self._node_name,
            service_proxy=[
                self._detect_bin_pose_client,
                self._get_bin_pose_client,
            ],
            action_clients=[],
        )

    def detect_bin_pose(
        self,
        bin_name: str,
        workspace_name: str,
        bin_mesh: neura_vision_tools_msgs.msg.Mesh = None,
        pointcloud_topic: Optional[str] = "/camera/depth_registered/points",
        method: Optional[int] = MethodType.WITHOUT_PREPROCESSING,
    ) -> Tuple[ReturnCode, ObjectWithPose]:
        """
        Detect bin pose. This function will start a detection from the current
        camera pose. The pose will be saved internally for next call. By default, the
        service check validity of input bin mesh. If the mesh is empty, it reads the mesh file of the bin given by
        bin name used for the bin detection

        Parameters
        ----------
        bin_mesh: neura_vision_tools_msgs.msg.Mesh
            The triangle mesh msg of the bin defined by NEURA Robotics
        bin_name : str
            The name of the bin
        workspace_name : str
            The name of the workspace, where the bin is located
        pointcloud_topic: str
            The name of pointcloud topic from 3D camera, by default /camera/depth_registered/points
        method : MethodType
             Mehtod Type: WITHOUT_PREPROCESSING to use pre-recorded workspace,
                          WITH_PREPROCESSING to use instance segmentaion based
                            approach, by default WITHOUT_PREPROCESSING.

        Returns
        -------
        Tuple[ReturnCode, ObjectWithPose]
            Return code and the pose of the bin.
        """

        pointcloud_msg = rospy.wait_for_message(
            pointcloud_topic,
            PointCloud2,
            timeout=3.0,
        )
        if pointcloud_msg is None:
            return (
                ReturnCode(
                    ReturnCodes.DATA_NOT_AVAILABLE, "no pointcloud msg received"
                ),
                None,
            )
        srv = DetectBinPoseRequest()
        if not bin_mesh is None:
            srv.bin_mesh = bin_mesh
        srv.bin_name = bin_name
        srv.scene_cloud = pointcloud_msg
        srv.workspace_name = workspace_name
        srv.method.input_type = method
        return_code, res = self._call_service(
            self._detect_bin_pose_client,
            srv,
            f"Detect pose for bin {bin_name} failed!",
        )
        if res is not None:
            return return_code, ObjectWithPose(
                bin_name,
                rc.geometry_msg_pose_2_pose(res.bin_pose.bin_pose_stamped.pose),
            )
        else:
            return return_code, None

    def get_bin_pose(self, bin_name: str) -> Tuple[ReturnCode, ObjectWithPose]:
        """
        Read the newest bin pose. If no bin detection has been performed, the
        recorded inital pose will be returned.
        Otherwise the newest detected pose will be returned. This function is only valid
        if performing from the internal pc

        Parameters
        ----------
        bin_name : str
            The name of the bin

        Returns
        -------
        Tuple[ReturnCode, ObjectWithPose]
            Return code and the pose of the bin.
        """
        srv = GetDetectedBinPoseRequest()
        srv.bin_name = bin_name
        return_code, res = self._call_service(
            self._get_bin_pose_client,
            srv,
            f"Get pose for bin {bin_name} failed!",
        )
        if res is not None:
            return return_code, ObjectWithPose(
                bin_name,
                rc.geometry_msg_pose_2_pose(res.bin_pose.bin_pose_stamped.pose),
            )
        else:
            return return_code, None

    def _initialize_clients(
        self,
        client_names: List[str],
        srv_types: List[genpy.Message],
        node_name: str,
    ) -> List[rospy.ServiceProxy]:
        return [
            rospy.ServiceProxy(
                "/" + node_name + "/" + client_name,
                srv_type,
            )
            for client_name, srv_type in zip(client_names, srv_types)
        ]

    def _call_service(
        self, client: rospy.ServiceProxy, srv: genpy.Message, warning: str
    ) -> Tuple[ReturnCode, Any]:
        try:
            res = client.call(srv)
            if res.return_code.value != ReturnCodes.SUCCESS:
                self._log.warning(warning)
                return (
                    ReturnCode(
                        ReturnCodes.SERVICE_CALL_RETURN_ERROR,
                        res.return_code.message,
                    ),
                    None,
                )
            self._log.debug("Service call returned")
            self._log.debug(
                "Response: %d %s"
                % (res.return_code.value, res.return_code.message)
            )
            return ReturnCode(), res
        except rospy.ServiceException as e:
            self._log.warn("Service call failed: %s" % e)
            return (
                ReturnCode(
                    ReturnCodes.SERVICE_CALL_FAILED, "Service call failed!"
                ),
                None,
            )
