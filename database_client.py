#!/usr/bin/env python3
"""Client to read the robot database."""

from typing import List, Tuple

import rospy
from neura_ai_database_msgs.srv import (
    ReadEndEffector,
    ReadEndEffectorRequest,
    ReadTCPPose,
    ReadTCPPoseRequest,
    ReadWorkspace,
    ReadWorkspaceRequest,
)
from std_srvs.srv import Trigger

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils.ros_conversions import geometry_msg_pose_2_pose
from neurapy_ai.utils.types import (
    EndEffector,
    JointState,
    Pose,
    TCPPose,
    Workspace,
)


class DatabaseClient(BaseAiClient):
    """Client to read the robot database.

    The database contains robot and workspace related information that are saved
    by the user via the robot HMI.

    Methods
    ----------
    get_pose
        Read robot position (point in database) as a robot pose.
    get_joint_position
        Read robot position (point in database) as joint states.
    get_workspace
        Read workspace stored in the database.
    get_end_effector
        Read end effector stored in the database.
    update_database
        Update information on the database server memory.
    """

    def __init__(self):
        """Initialize client."""
        self._node_name = "neura_ai_database"
        self._point_proxy = rospy.ServiceProxy(
            "/" + self._node_name + "/read_tcpPose", ReadTCPPose
        )
        self._workspace_proxy = rospy.ServiceProxy(
            "/" + self._node_name + "/read_workspace", ReadWorkspace
        )
        self._end_effector_proxy = rospy.ServiceProxy(
            "/" + self._node_name + "/read_end_effector", ReadEndEffector
        )
        self._update_database_proxy = rospy.ServiceProxy(
            "/" + self._node_name + "/update_database", Trigger
        )
        service_proxies = [
            self._point_proxy,
            self._workspace_proxy,
            self._end_effector_proxy,
            self._update_database_proxy,
        ]
        super(DatabaseClient, self).__init__(
            self._node_name,
            service_proxies,
            [],
            has_parameters=False,
        )
        self.update_database()

    def get_pose(self, point_name: str) -> Tuple[ReturnCode, Pose]:
        """Read robot position (point in database) as a robot pose.

        Parameters
        ----------
        point_name : str
            The name of a registered point

        Returns
        -------
        ReturnCode
            Return code
        Pose
            Pose in robot coordinate frame
        """
        return_code, _, point = self._read_tcp_pose(point_name)
        return return_code, point

    def get_joint_positions(
        self, point_name: str
    ) -> Tuple[ReturnCode, List[float]]:
        """Read robot position (point in database) as joint states.

        Parameters
        ----------
        point_name : str
            The name of a registered point

        Returns
        -------
        ReturnCode
            Return code
        List[float]
            Joint states

        """
        return_code, joint_states, _ = self._read_tcp_pose(point_name)
        return return_code, list(joint_states)

    def get_workspace(
        self, workspace_name: str
    ) -> Tuple[ReturnCode, Workspace]:
        """Read workspace stored in the database.

        Parameters
        ----------
        workspace_name : str
            The name of a registered workspace

        Returns
        -------
        ReturnCode
            Return code
        Workspace
            The workspace
        """
        self._wait_for_service_and_raise(self._workspace_proxy)
        req = ReadWorkspaceRequest()
        req.workspace_name = workspace_name

        try:
            response = self._workspace_proxy(req)
            if response.return_code.value < 0:
                self._log.error(response.return_code.message)
                return (
                    ReturnCode(
                        response.return_code.value,
                        response.return_code.message,
                    ),
                    None,
                )
        except rospy.ServiceException as e:
            self._log.error("Service call failed: %s", e)
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), None
        ws = response.workspace
        return ReturnCode(), Workspace(
            pose=geometry_msg_pose_2_pose(ws.transform_workspace2ref),
            frame=ws.ref_frame,
            len_x=ws.x_max,
            len_y=ws.y_max,
            len_z=ws.z_max,
            lookat_poses=[
                TCPPose(
                    geometry_msg_pose_2_pose(point.transform_tcp2ref),
                    JointState(point.tcp_pose_joint_space),
                )
                for point in ws.lookat_points
            ],
            name=workspace_name,
            type="tabletop" if ws.type == ws.TABLETOP else "bin",
            mesh_model=ws.mesh_model,
            collision_padding=ws.collision_padding,
        )

    def get_end_effector(
        self, end_effector_name: str = ""
    ) -> Tuple[ReturnCode, EndEffector]:
        """Read end effector stored in the database.

        Parameters
        ----------
        end_effector_name : str
            The name of a registered end effector. Defaults to "", to get the
            currently selected end effector.

        Returns
        -------
        ReturnCode
            Return code
        EndEffector
            The end effector
        """
        self._wait_for_service_and_raise(self._end_effector_proxy)
        req = ReadEndEffectorRequest()
        req.end_effector_name = end_effector_name

        try:
            response = self._end_effector_proxy(req)
            if response.return_code.value < 0:
                self._log.error(response.return_code.message)
                return (
                    ReturnCode(
                        response.return_code.value,
                        response.return_code.message,
                    ),
                    None,
                )
        except rospy.ServiceException as e:
            self._log.error("Service call failed: %s", e)
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), None
        tcp_pose = geometry_msg_pose_2_pose(
            response.end_effector.transform_tcp2flange
        )
        return ReturnCode(), EndEffector(
            name=response.end_effector_name,
            neura_typename=response.neura_supported_typename,
            tcp_pose=tcp_pose,
        )

    def update_database(self) -> ReturnCode:
        """
        Update information on the database server memory.

        Returns
        -------
        ReturnCode
            Return code
        """
        self._wait_for_service_and_raise(self._update_database_proxy)
        try:
            response = self._update_database_proxy()
            if not response.success:
                self._log.error(response.message)
                return ReturnCode(False, response.message)
        except rospy.ServiceException as e:
            self._log.error("Service call failed: %s", e)
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)

        return ReturnCode(True, response.message)

    def _read_tcp_pose(self, point_name):
        self._wait_for_service_and_raise(self._point_proxy)
        req = ReadTCPPoseRequest()
        req.tcp_point_name = point_name

        try:
            response = self._point_proxy(req)
            if response.return_code.value < 0:
                self._log.error(response.return_code.message)
                return (
                    ReturnCode(
                        response.return_code.value, response.return_code.message
                    ),
                    [],
                    None,
                )
        except rospy.ServiceException as e:
            self._log.error("Service call failed: %s", e)
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), [], None
        return (
            ReturnCode(),
            response.tcp_pose.tcp_pose_joint_space,
            geometry_msg_pose_2_pose(response.tcp_pose.transform_tcp2ref),
        )
