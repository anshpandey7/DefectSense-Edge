"""
Image folder frame capture source.
Supports reading existing image folders or automatically generating realistic
synthetic steel plate samples when run in offline/demo mode without datasets.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.capture.base import FrameSource


def generate_synthetic_plate(defect_type: str | None = None, width: int = 640, height: int = 640) -> np.ndarray:
    """Generate a realistic textured hot-rolled steel surface sample."""
    # Base metallic gray texture with slight grain
    base_color = 125 + np.random.randint(-15, 15)
    img = np.full((height, width), base_color, dtype=np.uint8)

    # Add rolling direction horizontal texture lines
    noise = np.random.normal(0, 10, (height, width)).astype(np.float32)
    # Stretch horizontally to simulate sheet rolling lines
    noise_smooth = cv2.GaussianBlur(noise, (31, 3), 0)
    img = np.clip(img.astype(np.float32) + noise_smooth, 0, 255).astype(np.uint8)

    # Convert to 3-channel BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if defect_type == "crack":
        # Draw realistic jagged fracture line
        points = []
        x, y = np.random.randint(40, 120), np.random.randint(40, 100)
        points.append((x, y))
        for _ in range(8):
            x += np.random.randint(15, 30)
            y += np.random.randint(-8, 14)
            points.append((x, y))
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(bgr, [pts], False, (20, 20, 25), 3, lineType=cv2.LINE_AA)
        cv2.polylines(bgr, [pts], False, (60, 60, 70), 1, lineType=cv2.LINE_AA)

    elif defect_type == "pitted_surface":
        # Cluster of dark corrosion pits
        cx, cy = np.random.randint(150, 450), np.random.randint(150, 450)
        for _ in range(40):
            px = int(cx + np.random.normal(0, 35))
            py = int(cy + np.random.normal(0, 35))
            r = np.random.randint(2, 6)
            cv2.circle(bgr, (px, py), r, (35, 35, 40), -1)

    elif defect_type == "inclusion":
        # Non-metallic dark particle inclusion with halo
        cx, cy = np.random.randint(200, 400), np.random.randint(200, 400)
        cv2.ellipse(bgr, (cx, cy), (28, 14), 30, 0, 360, (30, 30, 35), -1)
        cv2.ellipse(bgr, (cx, cy), (32, 17), 30, 0, 360, (190, 190, 195), 1)

    elif defect_type == "scratches":
        # Longitudinal fine scratches
        for _ in range(3):
            sy = np.random.randint(100, 500)
            cv2.line(bgr, (50, sy), (550, sy + np.random.randint(-10, 10)), (210, 210, 215), 1, lineType=cv2.LINE_AA)
            cv2.line(bgr, (52, sy + 1), (552, sy + 1 + np.random.randint(-10, 10)), (40, 40, 45), 1, lineType=cv2.LINE_AA)

    elif defect_type == "patches":
        # Localized irregular surface oxidation patch
        cx, cy = np.random.randint(200, 400), np.random.randint(200, 400)
        overlay = bgr.copy()
        cv2.ellipse(overlay, (cx, cy), (65, 45), 15, 0, 360, (70, 75, 80), -1)
        cv2.addWeighted(overlay, 0.6, bgr, 0.4, 0, bgr)

    elif defect_type == "crazing":
        # Network of micro fissures
        for _ in range(12):
            x1 = np.random.randint(100, 300)
            y1 = np.random.randint(100, 300)
            x2 = x1 + np.random.randint(-20, 20)
            y2 = y1 + np.random.randint(-20, 20)
            cv2.line(bgr, (x1, y1), (x2, y2), (30, 30, 30), 1)

    return bgr


class ImageFolderSource(FrameSource):
    """
    Reads an image folder continuously or stepped.
    If the folder is empty or non-existent, it auto-generates sample test plates.
    """

    def __init__(self, folder_path: str | Path, fps: int = 10, loop: bool = True, auto_generate_samples: bool = True):
        self.folder_path = Path(folder_path)
        self.fps = fps
        self.loop = loop
        self.auto_generate_samples = auto_generate_samples
        self.image_paths: List[Path] = []
        self.index = 0
        self.frame_delay = 1.0 / max(1, fps)
        self.last_frame_time = 0.0

    def open(self) -> bool:
        self.folder_path.mkdir(parents=True, exist_ok=True)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_paths = sorted([p for p in self.folder_path.glob("*") if p.suffix.lower() in extensions])

        if not self.image_paths and self.auto_generate_samples:
            # Generate initial sample plates representing different defect classes + clean
            samples = [
                ("plate_001_clean.jpg", None),
                ("plate_002_crack.jpg", "crack"),
                ("plate_003_inclusion.jpg", "inclusion"),
                ("plate_004_scratches.jpg", "scratches"),
                ("plate_005_pitted.jpg", "pitted_surface"),
                ("plate_006_patches.jpg", "patches"),
                ("plate_007_crazing.jpg", "crazing"),
                ("plate_008_clean.jpg", None),
            ]
            for fname, defect in samples:
                p = self.folder_path / fname
                img = generate_synthetic_plate(defect)
                cv2.imwrite(str(p), img)
                self.image_paths.append(p)

        self.index = 0
        return len(self.image_paths) > 0

    def read(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        if not self.image_paths:
            return False, None, {}

        # Respect target FPS timing
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed < self.frame_delay:
            time.sleep(max(0.001, self.frame_delay - elapsed))
        self.last_frame_time = time.time()

        if self.index >= len(self.image_paths):
            if self.loop:
                self.index = 0
            else:
                return False, None, {}

        img_path = self.image_paths[self.index]
        self.index += 1
        frame = cv2.imread(str(img_path))
        if frame is None:
            return False, None, {"error": f"Failed to read {img_path}"}

        metadata = {
            "source_path": str(img_path),
            "filename": img_path.name,
            "frame_index": self.index,
            "total_files": len(self.image_paths),
        }
        return True, frame, metadata

    def release(self) -> None:
        self.image_paths.clear()
        self.index = 0

    @property
    def source_type(self) -> str:
        return "folder"
