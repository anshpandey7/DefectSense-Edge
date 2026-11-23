"""
Hardware detection and backend selection module for DefectSense Edge.
Detects Raspberry Pi 5, Arm Cortex-A76, Hailo NPU, and CPU fallback.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

BackendType = Literal["cpu_onnx", "hailo"]


@dataclass
class HardwareProfile:
    os_name: str
    machine: str
    processor: str
    cpu_count: int
    is_arm: bool
    is_raspberry_pi: bool
    pi_model: str | None
    hailo_detected: bool
    hailo_device_info: str | None
    selected_backend: BackendType
    ram_gb: float

    def summary(self) -> str:
        pi_str = f"Raspberry Pi: {self.pi_model}" if self.is_raspberry_pi else "Non-Pi Host"
        hailo_str = f"Hailo NPU: DETECTED ({self.hailo_device_info})" if self.hailo_detected else "Hailo NPU: Not present"
        return (
            f"[HardwareProfile] OS: {self.os_name} {self.machine} | CPU Cores: {self.cpu_count} "
            f"| RAM: {self.ram_gb:.1f} GB | {pi_str} | {hailo_str} | Vision Backend: {self.selected_backend}"
        )


def _check_raspberry_pi() -> tuple[bool, str | None]:
    """Check if host is a Raspberry Pi and determine model."""
    # Method 1: Check device tree model
    dt_path = Path("/proc/device-tree/model")
    if dt_path.exists():
        try:
            model_text = dt_path.read_text(encoding="utf-8").strip("\x00 \n\r")
            if "Raspberry Pi" in model_text:
                return True, model_text
        except Exception:
            pass

    # Method 2: Check /proc/cpuinfo
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            text = cpuinfo.read_text(encoding="utf-8")
            if "Raspberry Pi" in text or "BCM2712" in text or "BCM2835" in text:
                for line in text.splitlines():
                    if line.startswith("Model"):
                        return True, line.split(":", 1)[1].strip()
                return True, "Raspberry Pi (Generic)"
        except Exception:
            pass

    return False, None


def _check_hailo_present() -> tuple[bool, str | None]:
    """
    Check if a Hailo-8 or Hailo-8L NPU (e.g., Raspberry Pi AI HAT+) is available.
    Checks /dev/hailo0 and pyhailort import.
    """
    # 1. Check devnode
    hailo_dev = Path("/dev/hailo0")
    if hailo_dev.exists():
        return True, "Hailo PCIe device node (/dev/hailo0) present"

    # 2. Check pyhailort Python SDK
    try:
        import hailo_platform  # type: ignore # noqa: F401
        return True, "HailoRT SDK module imported"
    except ImportError:
        pass

    # 3. Environment override for testing/simulation
    if os.environ.get("DEFECTSENSE_FORCE_HAILO", "0") == "1":
        return True, "Forced by DEFECTSENSE_FORCE_HAILO env var"

    return False, None


def get_ram_gb() -> float:
    """Get system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        # Fallback reading /proc/meminfo if on linux
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                for line in meminfo.read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
            except Exception:
                pass
        return 8.0  # Default estimate for Pi 5 8GB


def detect_hardware(preferred_backend: str = "auto") -> HardwareProfile:
    """
    Auto-detect system hardware and pick the best vision backend.
    """
    os_name = platform.system()
    machine = platform.machine()
    processor = platform.processor() or "Unknown"
    cpu_count = os.cpu_count() or 4
    is_arm = machine.lower() in ("aarch64", "arm64", "armv7l", "armv8l")

    is_pi, pi_model = _check_raspberry_pi()
    hailo_detected, hailo_info = _check_hailo_present()
    ram_gb = get_ram_gb()

    # Backend selection logic:
    # If preferred is explicitly given and valid:
    if preferred_backend == "hailo":
        if hailo_detected:
            selected_backend: BackendType = "hailo"
        else:
            logger.warning("Hailo backend was requested but no Hailo hardware detected! Falling back to cpu_onnx.")
            selected_backend = "cpu_onnx"
    elif preferred_backend == "cpu_onnx":
        selected_backend = "cpu_onnx"
    else:  # "auto"
        if hailo_detected:
            selected_backend = "hailo"
        else:
            selected_backend = "cpu_onnx"

    profile = HardwareProfile(
        os_name=os_name,
        machine=machine,
        processor=processor,
        cpu_count=cpu_count,
        is_arm=is_arm,
        is_raspberry_pi=is_pi,
        pi_model=pi_model,
        hailo_detected=hailo_detected,
        hailo_device_info=hailo_info,
        selected_backend=selected_backend,
        ram_gb=ram_gb,
    )
    logger.info(profile.summary())
    return profile
