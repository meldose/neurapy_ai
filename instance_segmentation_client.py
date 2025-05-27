#!/usr/bin/env python3
"""Client to segment object instances or train instance segmentation models."""

import signal
from typing import List, Optional, Tuple, Sequence

import actionlib
import numpy as np
import numpy.typing as npt
import rospy
from cv_bridge import CvBridge

from instance_segmentation_ros_msgs.msg import (
    TrainModelAction,
    TrainModelFeedback,
    TrainModelGoal,
    TrainModelResult,
)
from instance_segmentation_ros_msgs.srv import (
    SegmentInstances,
    SegmentInstancesImage,
)
from neura_common_msgs.srv import GetModel, SelectModel, GetMethod, SelectMethod
from neura_vision_tools_py.utils.segmentation_visualization import Visualizer

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes
from neurapy_ai.utils.types import BoundingBox, SegmentedInstance


class InstanceSegmentationClient(BaseAiClient):
    """Client to segment object instances or train instance segmentation models.

    Object instances could be segmented using a trained segmentation model or a
    new instance segmentation model can be trained on a `neura-style` dataset.

    Methods
    -------
    set_model
        Load a new instance segmentation model.
    get_model
        Get the currently loaded instance segmentation model.
    set_method
        Load a new instance segmentation method.
    get_method
        Get the currently used instance segmentation method.
    get_segmented_instances
        Get segmented object instances.
    get_segmented_instances_from_image
        Get segmented object instances in a given image.
    visualize_segmentation_result
        Visualize a segmentation result in the input image.
    train_segmentation_model
        Train a new segmentation model with `neura-style` dataset/s.

    Notes
    -----
    - See `neurapy_ai.clients.DataGenerationClient` to generate your own
    `neura-style` dataset.

    """

    def __init__(
        self,
        model_name: Optional[str] = "",
        model_version: Optional[str] = "newest",
        training: Optional[bool] = False,
    ):
        """Initialize client.

        Parameters
        ----------
        model_name : str, optional
            Name of an instance segmentation model to load at startup, by
            default ''.
        model_version : str, optional
            Version of the instance segmentation model to load, by default
            'newest'.
        training: str, optional
            Flag whether it is allowed to use this client for training new
            segmentation models, by default False.
        """
        self._instance_segmentation_proxy = rospy.ServiceProxy(
            "/instance_segmentation/segment_instances", SegmentInstances
        )
        self._instance_segmentation_image_proxy = rospy.ServiceProxy(
            "/instance_segmentation/segment_instances_image",
            SegmentInstancesImage,
        )
        self._select_model_proxy = rospy.ServiceProxy(
            "/instance_segmentation/select_model", SelectModel
        )
        self._get_model_proxy = rospy.ServiceProxy(
            "/instance_segmentation/get_model", GetModel
        )
        self._select_method_proxy = rospy.ServiceProxy(
            "/instance_segmentation/select_method", SelectMethod
        )
        self._get_method_proxy = rospy.ServiceProxy(
            "/instance_segmentation/get_method", GetMethod
        )

        service_proxies = [
            self._instance_segmentation_proxy,
            self._instance_segmentation_image_proxy,
            self._select_model_proxy,
            self._get_model_proxy,
            self._select_method_proxy,
            self._get_method_proxy,
        ]
        self.node_name = "instance_segmentation"
        self._training_flag = training
        if self._training_flag:
            self._train_instance_segmentation_action_client = (
                actionlib.SimpleActionClient(
                    "/train_instance_segmentation/train_model", TrainModelAction
                )
            )
            action_clients = [self._train_instance_segmentation_action_client]

        super(InstanceSegmentationClient, self).__init__(
            self.node_name,
            service_proxies,
            action_clients if self._training_flag else [],
            has_parameters=True,
        )

        if model_name != "":
            self.set_model(model_name, model_version)

        self.bridge = CvBridge()

    def set_model(
        self, model_name: str, model_version: Optional[str] = "newest"
    ) -> ReturnCode:
        """Load a new instance segmentation model.

        Parameters
        ----------
        model_name : str
            Name of the instance segmentation model to load
        model_version : str, optional
            Version of the instance segmentation model that should be loaded, by
            default is 'newest'.

        Returns
        -------
        ReturnCode
            Numerical return code with error message

        """
        try:
            result = self._select_model_proxy(model_name, model_version)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for setting the instance segmentation model "
                + "failed: %s",
                e,
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)
        return result.return_code

    def get_model(self) -> Tuple[str, str, ReturnCode]:
        """Get the currently loaded instance segmentation model.

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
                "Service call for getting the instance segmentation model "
                + "failed: %s",
                e,
            )
            return "", "", ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)
        return result.model_name, result.model_version, result.return_code

    def set_method(self, method: str) -> ReturnCode:
        """Load a new instance segmentation method.


        Parameters
        ----------
        method : str
            Name of the instance segmentation method to load

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        """
        try:
            result = self._select_method_proxy(method)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for setting the instance segmentation method "
                f"failed: {e}"
            )
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)
        return result.return_code

    def get_method(self) -> Tuple[str, ReturnCode]:
        """Get the current instance segmentation method.

        Returns
        -------
        str
            Name of the current instance segmentation method
        ReturnCode
            Numerical return code with error message

        """
        try:
            result = self._get_method_proxy()
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for getting the instance segmentation method "
                f"failed: {e}"
            )
            return "", "", ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e)
        return result.method, result.return_code

    def get_segmented_instances(
        self, class_names: Optional[Sequence[str]] = ()
    ) -> Tuple[
        ReturnCode,
        List[SegmentedInstance],
        npt.NDArray[np.uint8],
        npt.NDArray[np.uint8],
    ]:
        """Get segmented object instances.

        Get instances for given object names that match.

        Parameters
        ----------
        class_names : Sequence[str], optional
            A list of object class names, by default () (all classes).

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        List[SegmentedInstance]
            List of segmented object instances
        numpy ndarray (dtype: uint8)
            Combined segmentation mask for all detected instances
        numpy ndarray (dtype: uint8)
            The input image into segmentation model

        """
        segmented_instances = []
        segmentation_mask = None
        try:
            result = self._instance_segmentation_proxy(class_names)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for instance segmentation failed: %s", e
            )
            return (
                ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e),
                segmented_instances,
                segmentation_mask,
                [],
                [],
            )

        return_code = ReturnCode()
        input_image = self.bridge.imgmsg_to_cv2(result.input_image)
        if result.return_code.value >= 0:
            for instance_msg in result.instances:
                bounding_box = BoundingBox(
                    instance_msg.bounding_box.min_x,
                    instance_msg.bounding_box.min_y,
                    instance_msg.bounding_box.max_x,
                    instance_msg.bounding_box.max_y,
                )

                instance = SegmentedInstance(
                    instance_msg.class_name,
                    instance_msg.segmentation_index,
                    instance_msg.detection_score,
                    bounding_box,
                    instance_msg.class_instance_index,
                    self.bridge.imgmsg_to_cv2(instance_msg.instance_mask),
                )

                segmented_instances.append(instance)
            segmentation_mask = self.bridge.imgmsg_to_cv2(
                result.segmentation_mask
            )
        else:
            self._log.error(
                "Calling instance segmentation failed: "
                + result.return_code.message
            )
        return_code.value = result.return_code.value
        return_code.message = result.return_code.message

        return return_code, segmented_instances, segmentation_mask, input_image

    def get_segmented_instances_from_image(
        self,
        color_image: npt.NDArray[np.uint8],
        class_names: Optional[Sequence[str]] = (),
    ) -> Tuple[ReturnCode, List[SegmentedInstance], npt.NDArray[np.uint8]]:
        """Get segmented object instances in a given image.

        Get all segmented objects in a given image or a filtered result based
        on given object names that match.

        Parameters
        ----------
        color_image : np.array (shape [height, width, 3], dtype: uint8)
            The input color image in RGB channel order and 0-255 value range
        class_names : Sequence[str], optional
            A list of object class names, by default () (all classes).

        Returns
        -------
        ReturnCode
            Numerical return code with error message
        List[SegmentedInstance]
            List of segmented object instances
        numpy ndarray (dtype: uint8)
            Combined segmentation mask for all detected instances
        """
        image = self.bridge.cv2_to_imgmsg(color_image, encoding="passthrough")

        segmented_instances = []
        segmentation_mask = None
        try:
            result = self._instance_segmentation_image_proxy(class_names, image)
        except rospy.ServiceException as e:
            self._log.error(
                "Service call for instance segmentation from image failed: %s",
                e,
            )
            return (
                ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, e),
                segmented_instances,
                segmentation_mask,
            )

        return_code = ReturnCode()
        if result.return_code.value >= 0:
            for instance_msg in result.instances:
                bounding_box = BoundingBox(
                    instance_msg.bounding_box.min_x,
                    instance_msg.bounding_box.min_y,
                    instance_msg.bounding_box.max_x,
                    instance_msg.bounding_box.max_y,
                )
                instance = SegmentedInstance(
                    instance_msg.class_name,
                    instance_msg.segmentation_index,
                    instance_msg.detection_score,
                    bounding_box,
                    instance_msg.class_instance_index,
                    instance_msg.instance_mask,
                )

                segmented_instances.append(instance)
            segmentation_mask = self.bridge.imgmsg_to_cv2(
                result.segmentation_mask
            )
        else:
            self._log.error(
                "Calling instance segmentation from image failed: "
                + result.return_code.message
            )
        return_code.value = result.return_code.value
        return_code.message = result.return_code.message
        return return_code, segmented_instances, segmentation_mask

    def visualize_segmentation_result(
        self,
        input_image: npt.NDArray[np.uint8],
        segmented_instances: List[SegmentedInstance],
        segmentation_mask: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """Visualize a segmentation result in the input image.

        Show bounding boxes, segmentation masks and predicted class names as
        labels.

        Parameters
        ----------
        input_image : numpy ndarray (dtype: uint8)
            Input image in RGB channel order, values in [0, 255]
        segmented_instances : List[SegmentedInstance]
            List of segmented object instances
        segmentation_mask : numpy ndarray (dtype: uint8)
            Combined segmentation mask for all detected instances

        Returns
        -------
        visualization : numpy ndarray (dtype: uint8)
            The visualization image

        """
        class_names = [instance.class_name for instance in segmented_instances]
        class_names = list(set(class_names))
        visualizer = Visualizer(class_names)
        visualization = visualizer.draw_predictions(
            input_image, segmented_instances, segmentation_mask
        )
        del visualizer

        return visualization

    def train_segmentation_model(
        self,
        model_name: str,
        dataset_names: List[str],
        dataset_types: List[str],
        method: str = "detectron2",
        num_iterations: Optional[int] = 1,
        warmup_iterations: Optional[int] = 0,
        learning_rate: Optional[float] = 0.0003,
        batch_size: Optional[int] = 1,
        checkpoint_iterations: Optional[int] = 1,
        filter_empty_images: bool = True,
        pretrained_model: Optional[str] = "",
    ) -> ReturnCode:
        r"""Train a new segmentation model with `neura-style` dataset/s.

        Please ensure you are in AI-Hub mode before calling this function. You
        may stop the training process with Ctrl+C. Using the same set of input
        parameters would resume a previously-stopped training process.

        Parameters
        ----------
        model_name : str
            Name of your new model
        dataset_names : List[str]
            Names of the datasets to be used to train your new model. Order
            should be in sync with arg: dataset_types. See example below
        dataset_types : List[str]
            Choice of dataset types to use for training, one of {"real",
            "synthetic"}. Order should be in sync with arg: dataset_names. See
            example below.
        method : str 
            The method to train your model for. 
            Current available methods are: ["detectron2"]
        num_iterations : int, optional
            Number of iterations (batches) to train the model, by default 1
        warmup_iterations : int, optional
            Total steps to linearly increase learning rate from 0 to the base
            learning rate, by default 0
        learning_rate : float, optional
            Base learning rate, by default 0.0003
        batch_size : int, optional
            Total samples in one batch (that each GPU sees; for multi-GPU
            machines), by default 1
        checkpoint_iterations : int, optional
            Number of iterations before saving a checkpoint, by default 1
        filter_empty_images : bool, optional
            If true, images without annotations are not considered during
            training, by default True
        pretrained_model : str, optional
            Name of existing trained model to be used as base for new model,
            by default ""

        Returns
        -------
        ReturnCode
            Numerical return code with error message

        Examples
        --------
        1. To train a model with both "real" and "synthetic" dataset types from
        a dataset named "X"

        >>> train_segmentation_model(model_name="my_model"\
        ...     dataset_names=["X", "X"],
        ...     dataset_types=["real", "synthetic"])

        """
        if not self._training_flag:
            err = (
                "This client instance does not allow model training. Please "
                + "re-instantiate the client with 'training=True'"
            )
            self._log.error(err)
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, err)

        goal = TrainModelGoal()
        goal.method = method
        goal.model_name = model_name
        goal.dataset_names = dataset_names
        goal.dataset_types = dataset_types
        goal.num_iterations = num_iterations
        goal.warmup_iterations = warmup_iterations
        goal.learning_rate = learning_rate
        goal.batch_size = batch_size
        goal.checkpoint_iterations = checkpoint_iterations
        goal.filter_empty_images = filter_empty_images
        goal.pretrained_model = pretrained_model

        def active_cb():
            self._log.info("Started training segmentation model...")

        def feedback_cb(feedback: TrainModelFeedback):
            progress = int(
                feedback.epochs_done / feedback.number_of_epochs * 100
            )
            self._log.info(f"Progress: {progress}%")

        # active_cb: Callable[[None], None] = lambda: self._log.info(
        #     "Started training segmentation model..."
        # )
        # feedback_cb: Callable[[TrainModelFeedback], None] = (
        #     lambda feedback: self._log.info(
        #         f"Progress: {int(feedback.epochs_done/feedback.number_of_epochs*100)}%"
        #     )
        # )
        # Start training
        try:
            self._train_instance_segmentation_action_client.send_goal(
                goal=goal, active_cb=active_cb, feedback_cb=feedback_cb
            )
        except rospy.ServiceException as e:
            self._log.error(f"Action call failed: {e}")
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, f"{e}")

        def custom_signal_handler(signum, frame):
            self._log.warning(
                "User initiated cancellation. Cancelling all goals..."
            )
            self._train_instance_segmentation_action_client.cancel_all_goals()
            msg = "User cancelled segmentation model training."
            raise KeyboardInterrupt(msg)

        signal.signal(signal.SIGINT, custom_signal_handler)
        try:
            self._train_instance_segmentation_action_client.wait_for_result()
        except KeyboardInterrupt as e:
            return ReturnCode(
                ReturnCodes.SERVICE_CALL_FAILED,
                f"{e}",
            )

        result: TrainModelResult = (
            self._train_instance_segmentation_action_client.get_result()
        )
        if result is None:
            self._log.warning("No results from server. Cancelling all goals...")
            self._train_instance_segmentation_action_client.cancel_all_goals()
            return ReturnCode(
                ReturnCodes.SERVICE_CALL_RETURN_ERROR,
                "No result from server.",
            )
        elif result.return_code.value != ReturnCodes.SUCCESS:
            self._log.error(result.return_code)
        return ReturnCode(result.return_code.value, result.return_code.message)
