"""
webcam_ai – Real-Time Webcam AI Object Detection package.

Public re-exports for convenient importing::

    from webcam_ai import Config, FPSCounter, Detection
"""

from webcam_ai.config import Config
from webcam_ai.detection import Detection
from webcam_ai.fps import FPSCounter

__all__ = [
    "Config",
    "Detection",
    "FPSCounter",
]

__version__ = "0.1.0"
