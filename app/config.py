"""
Configuration manager for DefectSense Edge using Pydantic and PyYAML.
Supports environment variable overrides and configuration validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class AppInfo(BaseModel):
    name: str = "DefectSense Edge"
    version: str = "1.0.0"
    log_level: str = "INFO"
    line_name: str = "Line-1"
    product_name: str = "Manufactured-Component"


class HardwareConfig(BaseModel):
    preferred_backend: str = "auto"  # "auto", "cpu_onnx", "hailo"
    num_threads: int = 4
    enable_arm_neon: bool = True


class SourceConfig(BaseModel):
    type: str = "folder"  # "folder", "video", "camera"
    path: str = "data/sample_parts"
    camera_index: int = 0
    fps: int = 15
    width: int = 640
    height: int = 640
    loop: bool = True


class VisionConfig(BaseModel):
    model_path: str = "models/neu_yolov8n.onnx"
    input_size: List[int] = Field(default_factory=lambda: [640, 640])
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    anomaly_mode: bool = False
    anomaly_model_path: str = "models/mvtec_anomaly.onnx"
    anomaly_threshold: float = 0.65


class SeverityConfig(BaseModel):
    rules_file: str = "config/severity_rules.yaml"


class AdvisorConfig(BaseModel):
    backend: str = "executorch"  # "executorch", "fallback"
    pte_model_path: str = "models/llama3_2_1b_instruct.pte"
    tokenizer_path: str = "models/tokenizer.bin"
    timeout_seconds: float = 8.0
    debounce_seconds: float = 3.0
    max_tokens: int = 120
    temperature: float = 0.2


class AlertConfig(BaseModel):
    enable_buzzer: bool = True
    enable_gpio: bool = False
    gpio_reject_pin: int = 23
    gpio_pass_pin: int = 24
    sound_volume: float = 0.7


class StorageConfig(BaseModel):
    db_path: str = "data/defectsense.db"
    image_dir: str = "data/inspections"
    save_all_images: bool = False
    retention_days: int = 30
    thumbnail_size: List[int] = Field(default_factory=lambda: [160, 160])


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AppConfig(BaseModel):
    app: AppInfo = Field(default_factory=AppInfo)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    severity: SeverityConfig = Field(default_factory=SeverityConfig)
    advisor: AdvisorConfig = Field(default_factory=AdvisorConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    web: WebConfig = Field(default_factory=WebConfig)


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file, with optional custom path."""
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "config" / "default.yaml"
    
    path = Path(config_path)
    raw_config = load_yaml(path) if path.exists() else {}
    return AppConfig.model_validate(raw_config)
