#!/usr/bin/env python3-
"""
Client for querying the known objects and segmentation / pose estimation models
"""

from typing import Dict, List, Tuple

import rospy

from neura_data_management_msgs.msg import (
    ContinuousSymmetry,
    ModelInfo,
    ModelVersionInfo,
    ObjectInfo,
    DatasetInfo,
    DatasetSize,
)
from neura_data_management_msgs.srv import (
    GetModelInfo,
    GetObjectInfo,
    GetDatasetInfo,
    Search,
)

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils import ros_conversions as conversions


class DataManagementClient(BaseAiClient):
    # TODO not sure if the name is good?
    """Client for retrieving information about known objects and segmentation
    / pose estimation models


    Methods
    -------

    get_segmentation_model_info
        Retrieve information about a given segmentation model
    get_pose_estimation_model_info
        Retrieve information about a given pose estimation model
    get_object_info
        Retrieve information about a given object
    get_dataset_info
        Retrieve information about a given dataset
    get_all_objects
        Return the names of all known objects
    get_all_segmentation_models
        Return the names of all known segmentation models
    get_all_pose_estimation_models
        Return the names of all known pose estimation models
    get_all_datasets
        Return the names of all known datasets
    get_segmentation_models_for_object
        Return the names of all known segmentation models that were trained
        to recognize the given object
    get_pose_estimation_models_for_object
        Return the names of all known pose estimation models that were
        trained for localizing the given object

    """

    def __init__(self):
        self._search_proxy = rospy.ServiceProxy("/data_manager/search", Search)
        self._get_object_info_proxy = rospy.ServiceProxy(
            "/data_manager/get_object_info", GetObjectInfo
        )
        self._get_model_info_proxy = rospy.ServiceProxy(
            "/data_manager/get_model_info", GetModelInfo
        )
        self._get_dataset_info_proxy = rospy.ServiceProxy(
            "/data_manager/get_dataset_info", GetDatasetInfo
        )

        service_proxies = [
            self._search_proxy,
            self._get_object_info_proxy,
            self._get_model_info_proxy,
            self._get_dataset_info_proxy,
        ]
        self.node_name = "data_manager"

        super(DataManagementClient, self).__init__(
            self.node_name, service_proxies, [], has_parameters=False
        )

    def get_segmentation_model_info(
        self,
        model_name: str,
        method: str,
    ) -> Tuple[ReturnCode, Dict]:
        """
        Retrieve information about a given segmentation model

        Parameters
        ----------
        model_name : str
            Name of the segmentation model
        method : str
            Method of the segmentation model

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        info_dict : Dict
            Dictionary with the model information

        """
        info_dict = {}
        try:
            response = self._get_model_info_proxy(
                model_name,
                "segmentation_model",
                method,
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting segmentation model info failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), info_dict

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            info_dict = self._model_info_msg_to_dict(response.info)
        else:
            self._log.error(
                "Get segmentation model info failed:"
                + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, info_dict

    def get_pose_estimation_model_info(
        self,
        model_name: str,
        method: str,
    ) -> Tuple[ReturnCode, Dict]:
        """
        Retrieve information about a given pose estimation model

        Parameters
        ----------
        model_name : str
            Name of the pose estimation model
        model_method : str
            Method of the pose estimation model

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        info_dict : Dict
            Dictionary with the model information

        """
        info_dict = {}
        try:
            response = self._get_model_info_proxy(
                model_name,
                "pose_model",
                method,
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting pose model info failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), info_dict

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            info_dict = self._model_info_msg_to_dict(response.info)
        else:
            self._log.error(
                "Get pose model info failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, info_dict

    def get_object_info(self, object_name: str) -> Tuple[ReturnCode, Dict]:
        """
        Retrieve information about a given object

        Parameters
        ----------
        object_name : str
            Name of the object

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        info_dict : Dict
            Dictionary with the object information

        """
        info_dict = {}
        try:
            response = self._get_object_info_proxy(object_name)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting object info failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), info_dict

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            info_dict = self._object_info_msg_to_dict(response.info)
        else:
            self._log.error(
                "Get object info failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, info_dict

    def get_dataset_info(self, dataset_name: str) -> Tuple[ReturnCode, Dict]:
        """
        Retrieve information about a given dataset

        Parameters
        ----------
        dataset_name : str
            Name of the dataset

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        info_dict : Dict
            Dictionary with the dataset information

        """
        info_dict = {}
        try:
            response = self._get_dataset_info_proxy(dataset_name)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting dataset info failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), info_dict

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            info_dict = self._dataset_info_msg_to_dict(response.info)
        else:
            self._log.error(
                "Get dataset info failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, info_dict

    def get_all_objects(self) -> Tuple[ReturnCode, List[str]]:
        """
        Return the names of all known objects

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        object_names : List[str]
            The names of all known objects

        """
        object_names = []
        try:
            response = self._search_proxy(
                name="",
                type="object",
                contains_type="",
                contains_name="",
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting all objects failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), object_names

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            object_names = [msg.name for msg in response.results]
        else:
            self._log.error(
                "Get all objects failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, object_names

    def get_all_segmentation_models(self) -> Tuple[ReturnCode, List[str]]:
        """
        Return the names of all known segmentation models

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        model_names : List[str]
            The names of all known segmentation models

        """
        model_names = []
        try:
            response = self._search_proxy(
                name="",
                type="segmentation_model",
                contains_type="",
                contains_name="",
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting all segmentation models failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), model_names

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            model_names = [msg.name for msg in response.results]
        else:
            self._log.error(
                "Get all segmentation models failed:"
                + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, model_names

    def get_all_pose_estimation_models(self) -> Tuple[ReturnCode, List[str]]:
        """
        Return the names of all known pose estimation models

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        model_names : List[str]
            The names of all known pose estimation models

        """
        model_names = []
        try:
            response = self._search_proxy(
                name="",
                type="pose_model",
                contains_type="",
                contains_name="",
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting all pose models failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), model_names

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            model_names = [msg.name for msg in response.results]
        else:
            self._log.error(
                "Get all pose models failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, model_names

    def get_all_datasets(self) -> Tuple[ReturnCode, List[str]]:
        """
        Return the names of all known datasets

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        dataset_names : List[str]
            The names of all known datsets

        """
        dataset_names = []
        try:
            response = self._search_proxy(
                name="",
                type="dataset",
                contains_type="",
                contains_name="",
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting all datasets failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), dataset_names

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            dataset_names = [msg.name for msg in response.results]
        else:
            self._log.error(
                "Get all objects failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, dataset_names

    def get_segmentation_models_for_object(
        self, object_name: str
    ) -> Tuple[ReturnCode, List[str]]:
        """
        Return the names of all known segmentation models that were trained to
        recognize the given object

        Parameters
        ----------
        object_name : str
            The name of the object

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        model_names : List[str]
            The names of all matching segmentation models

        """
        model_names = []
        try:
            response = self._search_proxy(
                name="",
                type="segmentation_model",
                contains_type="object",
                contains_name=object_name,
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting segmentation models for object "
                f" {object_name} failed: {e}"
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), model_names

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            model_names = [msg.name for msg in response.results]
        else:
            self._log.error(
                "Get segmentations models for object failed:"
                + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, model_names

    def get_pose_estimation_models_for_object(
        self, object_name: str
    ) -> Tuple[ReturnCode, List[str]]:
        """
        Return the names of all known pose estimation models that were trained
        for localizing the given object

        Parameters
        ----------
        object_name : str
            The name of the object

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        model_names : List[str]
            The names of all matching pose estimation models

        """
        model_names = []
        try:
            response = self._search_proxy(
                name="",
                type="pose_model",
                contains_type="object",
                contains_name=object_name,
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting segmentation models for object "
                f" {object_name} failed: {e}"
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e), model_names

        return_code = ReturnCode()
        if response.return_code.value >= 0:
            model_names = [msg.name for msg in response.results]
        else:
            self._log.error(
                "Get pose models for object failed:"
                + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code, model_names

    @staticmethod
    def _object_info_msg_to_dict(msg: ObjectInfo) -> Dict:
        # TODO Not sure if giving the symmetry info to the users makes sense,
        # especially in this form
        symmetries_discrete = [
            conversions.geometry_msg_transform_2_transformation_matrix(sym)
            for sym in msg.symmetries_discrete
        ]
        symmetries_discrete = list(
            map(lambda sym: sym.reshape(-1).tolist(), symmetries_discrete)
        )
        symmetries_continuous = [
            DataManagementClient._continuous_symmetry_msg_to_dict(sym)
            for sym in msg.symmetries_continuous
        ]

        size = {
            "diameter": msg.size.diameter,
            "min_x": msg.size.min.x,
            "min_y": msg.size.min.y,
            "min_z": msg.size.min.z,
            "size_x": msg.size.size.x,
            "size_y": msg.size.size.y,
            "size_z": msg.size.size.z,
        }

        out_dict = {
            "name": msg.name,
            "is_scanned": msg.is_scanned,
            "is_symmetric": len(symmetries_discrete)
            + len(symmetries_continuous)
            > 0,
            "symmetries_discrete": symmetries_discrete,
            "symmetries_continuous": symmetries_continuous,
            "mesh_unit": msg.mesh_unit,
            "size": size,
        }

        return out_dict

    @staticmethod
    def _model_info_msg_to_dict(msg: ModelInfo) -> Dict:
        versions = [
            DataManagementClient._model_version_msg_to_dict(version)
            for version in msg.versions
        ]
        out_dict = {
            "name": msg.name,
            "datasets": msg.datasets,
            "objects": msg.objects,
            "versions": versions,
        }

        return out_dict

    @staticmethod
    def _continuous_symmetry_msg_to_dict(msg: ContinuousSymmetry) -> Dict:
        out = {
            "axis": [msg.axis.x, msg.axis.y, msg.axis.z],
            "offset": [msg.offset.x, msg.offset.y, msg.offset.z],
        }
        return out

    @staticmethod
    def _model_version_msg_to_dict(msg: ModelVersionInfo) -> Dict:
        out = {
            "version": msg.version,
            "size_bytes": msg.size_bytes,
            "last_modified": msg.last_modified,  # ,
            # TODO since these fields are not working properly for segmentation
            # models, I would leave them out for now
            # "average_test_loss": msg.average_test_loss,
            # "test_losses": [
            #     {
            #         "dataset_name": loss_msg.dataset_name,
            #         "test_loss": loss_msg.test_loss,
            #     }
            #     for loss_msg in msg.test_losses
            # ],
        }
        return out

    @staticmethod
    def _dataset_info_msg_to_dict(msg: DatasetInfo) -> Dict:
        out_dict = {
            "name": msg.name,
            "total_size": DataManagementClient._dataset_size_msg_to_dict(
                msg.total_size
            ),
            "synthetic_size": DataManagementClient._dataset_size_msg_to_dict(
                msg.synthetic_size
            ),
            "real_size": DataManagementClient._dataset_size_msg_to_dict(
                msg.real_size
            ),
            "subsets": msg.subsets,
            "objects": msg.objects,
        }

        return out_dict

    @staticmethod
    def _dataset_size_msg_to_dict(msg: DatasetSize) -> Dict:
        out = {
            "num_images": {
                "train": msg.num_images_train,
                "val": msg.num_images_val,
                "test": msg.num_images_test,
            },
            "num_annotations": {
                "train": msg.num_annotations_train,
                "val": msg.num_annotations_val,
                "test": msg.num_annotations_test,
            },
        }
        return out
