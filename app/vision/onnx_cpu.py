"""
ONNX Runtime CPU Vision Backend for DefectSense Edge.
Runs exported YOLOv8n / YOLO11n surface defect detection models on CPU (ARM Cortex-A76 / x86_64).
Includes letterbox preprocessing, NMS postprocessing, and coordinate rescaling.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

from app.vision.backend_base import BaseDetector, Detection

logger = logging.getLogger(__name__)

DEFAULT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
    "crack",
]


def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_up: bool = True,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resize and pad image while meeting stride-multiple constraints.
    Returns:
        (padded_image, scale_ratio, (pad_left, pad_top))
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class ONNXCPUDetector(BaseDetector):
    """
    High-performance CPU inference backend using ONNX Runtime.
    Optimized for Raspberry Pi 5 Arm Cortex-A76 cores with multi-threading and FP32/INT8.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        class_names: Optional[List[str]] = None,
        num_threads: int = 4,
    ):
        super().__init__(model_path, confidence_threshold, iou_threshold)
        self.input_size = input_size
        self.class_names = class_names or DEFAULT_CLASSES
        self.num_threads = num_threads
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_names: List[str] = []

    @property
    def backend_name(self) -> str:
        return "cpu_onnx"

    def load(self) -> None:
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create session with CPUExecutionProvider
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Check input shape if fixed
        in_shape = inputs[0].shape
        if len(in_shape) == 4 and isinstance(in_shape[2], int) and isinstance(in_shape[3], int):
            self.input_size = (in_shape[2], in_shape[3])

        self.is_loaded = True
        logger.info(
            f"Loaded ONNX model: {model_path.name} (input: {self.input_name} {self.input_size}, "
            f"threads: {self.num_threads})"
        )

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Preprocess BGR frame to normalized NCHW tensor with letterboxing."""
        # 1. Letterbox to model input dimensions
        padded, ratio, (pad_w, pad_h) = letterbox(frame, self.input_size)

        # 2. BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # 3. HWC to CHW and normalize to [0, 1]
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0

        # 4. Add batch dimension -> (1, 3, H, W)
        tensor = np.expand_dims(tensor, axis=0)
        return tensor, ratio, (pad_w, pad_h)

    def predict(self, frame: np.ndarray) -> Tuple[List[Detection], float]:
        if not self.is_loaded or self.session is None:
            self.load()

        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        # 1. Preprocess
        tensor, ratio, (pad_w, pad_h) = self.preprocess(frame)

        # 2. ONNX Runtime inference
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        raw_output = outputs[0]  # shape typically (1, 4 + num_classes, num_anchors) or (1, N, 6)

        # 3. Postprocess
        detections = self._postprocess(raw_output, orig_w, orig_h, ratio, pad_w, pad_h)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        return detections, inference_ms

    def _postprocess(
        self,
        output: np.ndarray,
        orig_w: int,
        orig_h: int,
        ratio: float,
        pad_w: float,
        pad_h: float,
    ) -> List[Detection]:
        """
        Parses raw model output and applies Non-Maximum Suppression (NMS).
        Handles YOLOv8/v11 format (1, 4 + classes, 8400) or direct detection tables.
        """
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            # YOLOv8 format: (1, 4 + classes, N) -> transpose to (1, N, 4 + classes)
            output = np.transpose(output, (0, 2, 1))

        preds = output[0]  # (N, features)
        num_features = preds.shape[1]

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        if num_features >= 5:
            # Check if output is [cx, cy, w, h, class_probs...]
            num_classes = num_features - 4
            for row in preds:
                cx, cy, w, h = row[:4]
                scores = row[4:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])

                if conf >= self.confidence_threshold:
                    # Convert letterboxed cx,cy,w,h to pixel x1,y1,x2,y2 in original image
                    x1 = (cx - w / 2.0 - pad_w) / ratio
                    y1 = (cy - h / 2.0 - pad_h) / ratio
                    x2 = (cx + w / 2.0 - pad_w) / ratio
                    y2 = (cy + h / 2.0 - pad_h) / ratio

                    # Clip to original image bounds
                    x1 = max(0, min(orig_w - 1, int(round(x1))))
                    y1 = max(0, min(orig_h - 1, int(round(y1))))
                    x2 = max(0, min(orig_w - 1, int(round(x2))))
                    y2 = max(0, min(orig_h - 1, int(round(y2))))

                    box_w = max(1, x2 - x1)
                    box_h = max(1, y2 - y1)

                    boxes.append([x1, y1, box_w, box_h])
                    confidences.append(conf)
                    class_ids.append(class_id)

        # Apply OpenCV NMS
        detections: List[Detection] = []
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.iou_threshold)
            total_image_area = float(orig_w * orig_h)
            edge_margin_x = orig_w * 0.05
            edge_margin_y = orig_h * 0.05

            if len(indices) > 0:
                for idx in indices.flatten():
                    bx, by, bw, bh = boxes[idx]
                    x1, y1, x2, y2 = bx, by, bx + bw, by + bh
                    cid = class_ids[idx]
                    class_name = self.class_names[cid] if cid < len(self.class_names) else f"class_{cid}"
                    conf = confidences[idx]
                    area_pct = (bw * bh) / total_image_area * 100.0

                    is_edge = (x1 <= edge_margin_x or x2 >= (orig_w - edge_margin_x) or
                               y1 <= edge_margin_y or y2 >= (orig_h - edge_margin_y))

                    detections.append(Detection(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        class_name=class_name,
                        confidence=round(conf, 4),
                        area_pct=round(area_pct, 3),
                        is_edge=is_edge,
                    ))

        return detections
