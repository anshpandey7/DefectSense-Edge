"""
Capture package for DefectSense Edge.
"""

from app.capture.base import FrameSource
from app.capture.folder_source import ImageFolderSource, generate_synthetic_plate
from app.capture.camera_source import CameraSource, VideoSource

__all__ = [
    "FrameSource",
    "ImageFolderSource",
    "generate_synthetic_plate",
    "CameraSource",
    "VideoSource",
]
