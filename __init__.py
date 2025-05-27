"""Clients to Neuron.AI servers."""

# ruff: noqa: F401

from .autonomous_pick_client import AutonomousPickClient
from .database_client import DatabaseClient
from .data_based_pick_client import DataBasedPickClient
from .data_management_client import DataManagementClient
from .instance_segmentation_client import InstanceSegmentationClient

from .marker_detection_client import MarkerDetectionClient
from .pointing_pick_client import PointingPickClient
from .pointing_point_detection_client import PointingPointDetectionClient
from .pose_estimation_client import PoseEstimationClient

from .robot_scan_client import RobotScanClient
from .voice_control_client import VoiceControlClient
from .data_generation_client import DataGenerationClient
from .audio_output_client import AudioOutputClient
from .real_data_collection_client import RealDataCollectionClient
