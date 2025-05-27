#!/usr/bin/env python3
"""Client to play audio files."""

from pathlib import Path

import rospy
import sounddevice as sd
import soundfile as sf
from std_msgs.msg import Bool

from neura_play_sounds.srv import PlaySound, PlaySoundRequest

from neurapy_ai.clients.base_ai_client import BaseAiClient
from neurapy_ai.utils.return_codes import ReturnCode, ReturnCodes


class AudioOutputClient(BaseAiClient):
    """Client to play audio files.

    Audio can pe played on the chosen output device for sound feedback or user
    interaction.

    Methods
    -------
    beep_negative
        Play negative beep audio on the robot.
    beep_neutral
        Play neutral beep audio on the robot.
    beep_neutral_double
        Play neutral double beep audio on the robot.
    beep_positive
        Play positive beep audio on the robot.
    play_audio
        Play an audio file locally or in the robot system.
    stop_all_audio
        Stop playing all audio in the robot system.
    test_audio
        Play test audio on the robot.
    """

    def __init__(self):
        """Initialize client."""
        self.audio_output_proxy_ = rospy.ServiceProxy("sound_player", PlaySound)
        node_name = "sound_player_server"
        super(AudioOutputClient, self).__init__(
            node_name, [self.audio_output_proxy_], [], has_parameters=False
        )
        self.stop_pub_ = rospy.Publisher(
            "/stop_audio_playback", Bool, latch=False, queue_size=1
        )

    def beep_negative(self, blocking: bool = True) -> ReturnCode:
        """Play negative beep audio on the robot.

        Parameters
        ----------
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends

        Returns
        -------
        ReturnCode
            Return code with message
        """
        request = PlaySoundRequest()
        request.wait = blocking
        request.beep_choice = request.NEGATIVE
        return self._call_service(request)

    def beep_neutral(self, blocking: bool = True) -> ReturnCode:
        """Play neutral beep audio on the robot.

        Parameters
        ----------
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends

        Returns
        -------
        ReturnCode
            Return code with message
        """
        request = PlaySoundRequest()
        request.wait = blocking
        request.beep_choice = request.NEUTRAL
        return self._call_service(request)

    def beep_neutral_double(self, blocking: bool = True) -> ReturnCode:
        """Play neutral double beep audio on the robot.

        Parameters
        ----------
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends

        Returns
        -------
        ReturnCode
            Return code with message
        """
        request = PlaySoundRequest()
        request.wait = blocking
        request.beep_choice = request.NEUTRAL_DOUBLE
        return self._call_service(request)

    def beep_positive(self, blocking: bool = True) -> ReturnCode:
        """Play positive beep audio on the robot.

        Parameters
        ----------
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends

        Returns
        -------
        ReturnCode
            Return code with message
        """
        request = PlaySoundRequest()
        request.wait = blocking
        request.beep_choice = request.POSITIVE
        return self._call_service(request)

    def play_audio(
        self, audio_file: str, blocking: bool = True, target: str = "robot"
    ) -> ReturnCode:
        """Play an audio file based on system choice.

        Parameters
        ----------
        audio_file : str
            Path to an audio file
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends
        target : str, optional
            'local' to play audio locally or 'robot' to play audio on the robot,
            by default `robot`

        Returns
        -------
        ReturnCode
            Return code with message
        """
        if target.lower() == "robot":
            return_code = self._play_audio_robot(audio_file, blocking=blocking)
        elif target.lower() == "local":
            return_code = self._play_audio_local(audio_file, blocking=blocking)
        else:
            msg = f"Invalid value: target='{target.lower()}'. Expected 'local' "
            "or 'robot'."
            self._log.error(msg)
            return_code = ReturnCode(
                value=ReturnCodes.INVALID_ARGUMENT, message=msg
            )
        return return_code

    def stop_all_audio(self) -> ReturnCode:
        """Stop playing all audio in the robot system."""
        self.stop_pub_.publish(1)

    def test_audio(self, blocking: bool = True) -> ReturnCode:
        """Play test audio on the robot.

        Parameters
        ----------
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends

        Returns
        -------
        ReturnCode
            Return code with message
        """
        request = PlaySoundRequest()
        request.wait = blocking
        request.beep_choice = request.TEST
        return self._call_service(request)

    def _call_service(self, request: PlaySoundRequest):
        """Play audio request

        Parameters
        ----------
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends

        Returns
        -------
        ReturnCode
            Return code with message
        """
        try:
            self.audio_output_proxy_(request)
        except rospy.ServiceException as ex:
            self._log.error(f"Service call for playing audio failed: {ex}")
            return ReturnCode(ReturnCodes.SERVICE_CALL_FAILED, ex)
        return ReturnCode()

    def _play_audio_local(self, audio_file: str, blocking: bool = True):
        """Play audio on the local machine.

        Parameters
        ----------
        audio_file : str
            Path to an audio file
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends

        Returns
        -------
        ReturnCode
            Return code with message
        """
        if not Path(audio_file).exists():
            message = f"Requested audio file {audio_file} does not exist, did "
            "you mean to play a file from the robot's file system? If so "
            "change 'target' variable to 'robot'."
            self._log.error(message)
            return ReturnCode(ReturnCodes.DATA_NOT_AVAILABLE, message)

        data, fs = sf.read(audio_file, dtype="float32")
        sd.play(data, fs)
        if blocking:
            sd.wait()
        return ReturnCode()

    def _play_audio_robot(self, audio_file: str, blocking: bool = True):
        """Play audio on the robot.

        Parameters
        ----------
        audio_file : str
            Path to an audio file
        blocking : bool, optional
            Blocks while audio is playing. Default is True, to block
            execution until audio playing ends

        Returns
        -------
        ReturnCode
            Return code with message
        """
        request = PlaySoundRequest()
        request.wait = blocking
        request.filename = audio_file
        return self._call_service(request)
