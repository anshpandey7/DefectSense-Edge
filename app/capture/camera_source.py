"""
Camera and video frame capture sources.
Supports USB V4L2 webcams, Raspberry Pi Camera (libcamera / picamera2 / V4L2), and video files.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from app.capture.base import FrameSource


class CameraSource(FrameSource):
    """
    OpenCV VideoCapture camera source with support for V4L2, libcamera, and USB webcams.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 640, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_idx = 0

    def open(self) -> bool:
        # On Linux/Pi, CAP_V4L2 is preferred; on Windows default is DSHOW or MSMF
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.frame_idx = 0
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        if self.cap is None or not self.cap.isOpened():
            return False, None, {}
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, {}
        self.frame_idx += 1
        meta = {
            "camera_index": self.camera_index,
            "frame_index": self.frame_idx,
            "timestamp": time.time(),
        }
        return True, frame, meta

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def source_type(self) -> str:
        return "camera"


class VideoSource(FrameSource):
    """
    Video file source with automatic looping and FPS throttling.
    """

    def __init__(self, video_path: str | Path, loop: bool = True):
        self.video_path = Path(video_path)
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30
        self.frame_idx = 0
        self.last_time = 0.0

    def open(self) -> bool:
        if not self.video_path.exists():
            return False
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps > 0 else 30
        self.frame_idx = 0
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        if self.cap is None or not self.cap.isOpened():
            return False, None, {}

        # Throttle to natural video fps
        delay = 1.0 / self.fps
        now = time.time()
        elapsed = now - self.last_time
        if elapsed < delay:
            time.sleep(max(0.001, delay - elapsed))
        self.last_time = time.time()

        ret, frame = self.cap.read()
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if not ret or frame is None:
                return False, None, {}

        self.frame_idx += 1
        meta = {
            "video_path": str(self.video_path),
            "frame_index": self.frame_idx,
            "timestamp": time.time(),
        }
        return True, frame, meta

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def source_type(self) -> str:
        return "video"
