"""
Abstract base class for frame capture sources (camera, video, folder).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple
import numpy as np


class FrameSource(abc.ABC):
    """Abstract frame capture source."""

    @abc.abstractmethod
    def open(self) -> bool:
        """Initialize or connect to the source."""
        pass

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Fetch next frame.
        Returns:
            (success, frame_bgr, metadata)
        """
        pass

    @abc.abstractmethod
    def release(self) -> None:
        """Release any hardware/file resources."""
        pass

    @property
    @abc.abstractmethod
    def source_type(self) -> str:
        """Return type string: camera, video, folder."""
        pass
