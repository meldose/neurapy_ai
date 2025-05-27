#!/usr/bin/env python3
"""Client for `neura-style` dataset generation."""

import signal
from typing import List, Tuple

import actionlib
import rospy

from neura_data_generation_msgs.msg import (
    DataGenerationAction,
    DataGenerationFeedback,
    DataGenerationGoal,
    DataGenerationResult,
)

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes


class DataGenerationClient(BaseAiClient):
    """Client for `neura-style` dataset generation.

    Features include synthetic dataset generation which builds a scene with
    user-configured objects, with random background textures and distractor
    objects.

    Methods
    -------
    generate_synthetic_dataset
        Generate a synthetic dataset of images that is auto-annotated.
    """

    def __init__(self):
        """Initialize client."""
        node_name = "data_generation"
        self._data_generation_action_client = actionlib.SimpleActionClient(
            node_name, DataGenerationAction
        )
        super(DataGenerationClient, self).__init__(
            node_name=node_name,
            service_proxy=[],
            action_clients=[self._data_generation_action_client],
            has_parameters=False,
        )

    def generate_synthetic_dataset(
        self,
        dataset_name: str,
        object_names: List[str],
        images_generated_count: int,
        camera_distance_range: Tuple[float, float],
        light_energy_range: Tuple[int, int],
    ) -> ReturnCode:
        """Generate a synthetic dataset of images that is auto-annotated.

        This dataset generaties requires that the objects of interest are
        available in a `neura-style` folder structure.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        object_names : List[str]
            Names of the user objects to include in the dataset
        images_generated_count : int
            Number of images to generate and annotate.
        camera_distance_range : Tuple[float, float]
            (Min,Max) camera distance in 'm', to the closest object of interest
        light_energy_range : Tuple[int, int]
            (Min,Max) light energy in 'Wm-2' if the light type is SUN, otherwise
            'W'. Light type is randomly chosen between [POINT, SUN, SPOT, AREA]

        Returns
        -------
        ReturnCode
            Numerical return code with error message

        Raises
        ------
        KeyboardInterrupt
            Received SIGINT signal during data generation, which stops the
            process
        """
        goal = DataGenerationGoal()
        goal.dataset_name = dataset_name
        goal.object_names = object_names
        goal.number_images = images_generated_count
        goal.camera_distance_min = camera_distance_range[0]
        goal.camera_distance_max = camera_distance_range[1]
        goal.light_energy_min = light_energy_range[0]
        goal.light_energy_max = light_energy_range[1]

        def active_cb():
            self._log.info("Started generating data...")

        def feedback_cb(feedback: DataGenerationFeedback):
            self._log.info(f"Progress: {feedback.progress_percentage}%")

        # active_cb: Callable[[None], None] = lambda: self._log.info(
        #     "Started generating data..."
        # )
        # feedback_cb: Callable[[DataGenerationFeedback], None] = (
        #     lambda feedback: self._log.info(
        #         f"Progress: {feedback.progress_percentage}%"
        #     )
        # )
        try:
            self._data_generation_action_client.send_goal(
                goal=goal, active_cb=active_cb, feedback_cb=feedback_cb
            )
        except rospy.ServiceException as e:
            self._log.error(f"Action call failed: {e}")
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, f"{e}")

        def custom_signal_handler(signum, frame):
            self._log.warning(
                "User initiated cancellation. Cancelling all goals..."
            )
            self._data_generation_action_client.cancel_all_goals()
            msg = "User cancelled data generation."
            raise KeyboardInterrupt(msg)

        signal.signal(signal.SIGINT, custom_signal_handler)
        try:
            self._data_generation_action_client.wait_for_result()
        except KeyboardInterrupt as e:
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, f"{e}")

        result: DataGenerationResult = (
            self._data_generation_action_client.get_result()
        )
        if result is None:
            self._log.warning("No results from server. Cancelling all goals...")
            self._data_generation_action_client.cancel_all_goals()
            return ReturnCode(
                ReturnCodes.SERVICE_CALL_RETURN_ERROR,
                "No result from server.",
            )
        elif result.return_code.value != ReturnCodes.SUCCESS:
            self._log.error(result.return_code)
        return ReturnCode(result.return_code.value, result.return_code.message)
