#!/usr/bin/env python3
"""
Client for capturing new images and saving them as a dataset
"""
import rospy
from enum import IntEnum

from neura_data_generation_msgs.srv import (
    SaveDataset,
    SaveImage,
    ClearData,
)

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes


class RealDataCollectionClient(BaseAiClient):
    """Client for capturing new images and saving them as a dataset

    Methods
    -------

    save_image
        Save an image from the robot camera
    save_dataset
        Create a dataset from the images collected
    clear_data
        Clears all images that haven't been saved as a dataset yet.

    """

    class DatasetTypes(IntEnum):
        """Recognized dataset types for RealDataCollectionClient"""

        TO_LABEL = 1
        """Images that **must** be annotated with objects of interest."""

        ENVIRONMENT = 2
        """Images containing **no** objects of interest, intended for capturing 
        environmental context."""

    def __init__(self):
        self._save_image_proxy = rospy.ServiceProxy(
            "/real_data_collection/save_image", SaveImage
        )
        self._save_dataset_proxy = rospy.ServiceProxy(
            "/real_data_collection/save_dataset", SaveDataset
        )
        self._clear_data_proxy = rospy.ServiceProxy(
            "/real_data_collection/clear_data", ClearData
        )

        service_proxies = [
            self._save_image_proxy,
            self._save_dataset_proxy,
            self._clear_data_proxy,
        ]
        self.node_name = "real_data_collection"

        super(RealDataCollectionClient, self).__init__(
            self.node_name, service_proxies, [], has_parameters=False
        )

    def save_image(self) -> ReturnCode:
        """Save an image from the robot camera

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message

        """
        try:
            response = self._save_image_proxy()
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for saving an image failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)

        return_code = ReturnCode()
        if response.return_code.value != ReturnCodes.SUCCESS:
            self._log.error(
                "Saving image failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code

    def save_dataset(
        self,
        dataset_type: DatasetTypes,
        dataset_name: str,
    ) -> ReturnCode:
        """Create a dataset from the images collected

        Parameters
        ----------
        dataset_type : DatasetTypes
            Type of dataset to create
        dataset_name : str
            Name of the dataset that will be created.

        Returns
        -------
        return_code : ReturnCode
            Numerical return code with error message
        """

        try:
            response = self._save_dataset_proxy(
                dataset_type=dataset_type, dataset_name=dataset_name
            )
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for saving a dataset failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)

        return_code = ReturnCode()
        if response.return_code.value != ReturnCodes.SUCCESS:
            self._log.error(
                "Saving dataset failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code

    def clear_data(self) -> ReturnCode:
        """Clears all images that haven't been saved as a dataset.

        Returns
        -------
        return_code: ReturnCode
            Numerical return code with error message
        """
        try:
            response = self._clear_data_proxy()
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for clearing images failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)

        return_code = ReturnCode()
        if response.return_code.value != ReturnCodes.SUCCESS:
            self._log.error(
                "Clearing images failed:" + response.return_code.message
            )
            return_code.value = response.return_code.value
            return_code.message = response.return_code.message
        return return_code
