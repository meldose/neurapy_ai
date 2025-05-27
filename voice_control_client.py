#!/usr/bin/env python3
"""Client for voice-based interaction."""

from typing import Optional, Tuple

import actionlib
import rospy
from std_msgs.msg import Bool, Int16, String

from neura_voice_control_msgs.msg import GetCommandAction, GetCommandGoal
from neura_voice_control_msgs.srv import TriggerCommand, TriggerCommandRequest

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes


class VoiceControlClient(BaseAiClient):
    """Client for voice-based interaction.

    Methods
    -------
    get_command
        Get the voice command. Command has to be triggered by "Hey Maira".
    get_command_with_trigger
        Get the results of command defined as an argument to this function.
    set_command
        Set detecting command and trigger with "Hey Maira" (non-blocking).
    get_last_command
        Get last command.
    get_return_command
        Start continuous detection of the voice commands (blocking).
    finish
        Check if a command has executed.
    """

    def __init__(self):
        """Initialize client."""
        # action
        self._action_name = "/neura_voice_control/get_commmand_action"
        self._action_client = actionlib.SimpleActionClient(
            self._action_name, GetCommandAction
        )

        # service
        self._service_name = "/neura_voice_control/get_commmand_service"
        self._service_client = rospy.ServiceProxy(
            self._service_name, TriggerCommand
        )

        # publisher
        self._pub_common_command = rospy.Publisher(
            "/common_command", Int16, queue_size=1
        )

        # subscriber
        self._sub_return_command = rospy.Subscriber(
            "/command", String, self._return_command_cb
        )
        self._sub_command_finished = rospy.Subscriber(
            "/neura_voice_control/command_finished",
            Bool,
            self._command_finished_cb,
        )
        self._sub_back_conv = rospy.Subscriber(
            "/neura_voice_control/back_to_conversation",
            Bool,
            self._back_conv_cb,
        )

        # setup variables.
        self._return_command = None
        self._last_command = ""
        self._command_finished = False

        # init
        super(VoiceControlClient, self).__init__(
            "respeaker_ros",
            [self._service_client],
            [self._action_client],
            False,
        )

    def get_command_with_trigger(self, command: str) -> Tuple[ReturnCode, str]:
        """Get the results of command defined as an argument to this function

        Parameters
        ----------
        command : str
            command that should be started. If command is not defined and then
            return code is returned
        similar : bool
            <description>

        Returns
        -------
        ReturnCode
            Return code
        str
            resulting command
        """
        try:
            self._action_client.wait_for_server()
        except (rospy.ServiceException, rospy.ROSException) as e:
            self._log.error("Service connection failed: %s" % e)
            return ReturnCode(ReturnCodes.SERVICE_NOT_AVAILABLE), []

        goal = GetCommandGoal()
        goal.command = command
        try:
            self._action_client.send_goal(goal)
            self._action_client.wait_for_result()
            response = self._action_client.get_result()
            if response.return_code.value < 0:
                self._log.error(response.return_code.message)
                return (
                    ReturnCode(
                        response.return_code.value, response.return_code.message
                    ),
                    [],
                )
            return (
                ReturnCode(
                    response.return_code.value, response.return_code.message
                ),
                response.message,
            )
        except rospy.ServiceException as e:
            self._log.error("Service call failed: %s" % e)
            return (
                ReturnCode(
                    ReturnCodes.SERVICE_CALL_FAILED, "Service call failed"
                ),
                "",
            )

    def get_command(
        self, timeout: Optional[float] = None
    ) -> Tuple[ReturnCode, str]:
        """Get the voice command. Command has to be triggered by "Hey Maira".

        Parameters
        ----------
        timeout : float, optional
            if specified funtion will terminate after that time even if no
            command was triggered, by default None - no timeout will be used

        Returns
        -------
        ReturnCode
            Return code
        str
            Resulting command (without actual command)
        """
        try:
            command = rospy.wait_for_message("command", String, timeout)
            return ReturnCode(ReturnCodes.SUCCESS, ""), command
        except rospy.ROSException as e:
            self._log.error("Command failed: %s" % e)
            return (
                ReturnCode(ReturnCodes.SERVICE_NOT_AVAILABLE, "command failed"),
                "",
            )

    def set_command(self, command: str, similar: bool) -> Tuple[bool, str]:
        """Set detecting command and trigger with "Hey Maira" (non-blocking).

        Parameters
        ----------
        command : str
            Command that should be started.
        similar : bool
            <description>

        Returns
        -------
        bool
            The return value, true if command setting success, false for
            otherwise.
        str
            Return message.
        """
        request = TriggerCommandRequest()
        request.command = command
        request.similar = similar
        try:
            self._service_client(request)
        except rospy.ServiceException as e:
            self._log.error("Service call failed: %s" % e)
            self._return_code = ReturnCode(
                ReturnCodes.SERVICE_CALL_FAILED, "Service call failed!"
            )
            return False, self._return_code
        return True, ""

    def get_last_command(self) -> Tuple[ReturnCode, str]:
        """Get last command.

        If there were no command since the last call to this function, returns
        empty string. start_continous_detection() has to be started before,
        otherwise FUNCTION_NOT_INITIALIZED error code is returned.

        Returns
        -------
        ReturnCode
            Return code
        str
            The last command
        """
        if self._return_command is None:
            ReturnCode(
                ReturnCodes.FUNCTION_NOT_INITIALIZED,
                "continous detection was not activated",
            ), ""
        return ReturnCode(), self._last_command

    def get_return_command(self) -> str:
        """Start continuous detection of the voice commands (blocking).

        Commands have to be started by "Hey Maira". Commands can be read by
        calling `get_last_command` function.

        Returns
        -------
        str
            The last command
        """
        while not self._last_command:
            if self._command_finished:
                break
            continue
        return_command = self._last_command
        self._reset_command()
        return return_command

    def finish(self) -> bool:
        """Check if a command has been executed.

        Returns
        -------
        bool
            True if command has been executed, else False
        """
        return_value = self._command_finished
        if self._command_finished:
            self._command_finished = False
        return return_value

    def _reset_command(self):
        self._last_command = ""

    def _return_command_cb(self, msg):
        self._last_command = msg.data

    def _command_finished_cb(self, msg):
        self._command_finished = msg.data

    def _back_conv_cb(self, msg):
        self._command_finished = msg.data
