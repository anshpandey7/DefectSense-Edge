"""
Industrial alert controller for DefectSense Edge.
Supports Raspberry Pi GPIO relays / indicator LEDs (BCM pins) and audible buzzers.
Falls back to Mock GPIO and system sound when running on non-Pi development hardware.
"""

from __future__ import annotations

import logging
import platform
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger("alerts.gpio")


class GPIOAlertController:
    """
    Manages physical indicators:
    - Reject relay / Red beacon (e.g., BCM pin 23)
    - Pass indicator / Green beacon (e.g., BCM pin 24)
    - Audible buzzer
    """

    def __init__(
        self,
        enable_gpio: bool = False,
        enable_buzzer: bool = True,
        reject_pin: int = 23,
        pass_pin: int = 24,
    ):
        self.enable_gpio = enable_gpio
        self.enable_buzzer = enable_buzzer
        self.reject_pin = reject_pin
        self.pass_pin = pass_pin

        self.gpio_lib = None
        self.is_real_gpio = False

        if self.enable_gpio:
            self._init_gpio()

    def _init_gpio(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.reject_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.pass_pin, GPIO.OUT, initial=GPIO.LOW)
            self.gpio_lib = GPIO
            self.is_real_gpio = True
            logger.info(f"Initialized physical RPi.GPIO on pins Reject={self.reject_pin}, Pass={self.pass_pin}")
        except Exception as e:
            logger.info(f"RPi.GPIO not active ({e}); running in simulated Mock GPIO mode.")
            self.is_real_gpio = False

    def trigger_verdict_alert(self, verdict: str, severity: str = "LOW") -> None:
        """
        Trigger indicators according to verdict.
        Runs asynchronously in a helper thread to prevent delaying video frames.
        """
        threading.Thread(target=self._pulse_outputs, args=(verdict, severity), daemon=True).start()

    def _pulse_outputs(self, verdict: str, severity: str) -> None:
        if verdict == "REJECT":
            # Pulse reject relay/red LED for 500ms
            if self.is_real_gpio and self.gpio_lib:
                self.gpio_lib.output(self.reject_pin, self.gpio_lib.HIGH)
                time.sleep(0.5)
                self.gpio_lib.output(self.reject_pin, self.gpio_lib.LOW)
            else:
                logger.debug(f"[MOCK GPIO] PULSE HIGH -> Reject Pin {self.reject_pin} (REJECT / {severity})")

            if self.enable_buzzer:
                self._sound_buzzer(freq=1200, duration_sec=0.25)

        elif verdict == "PASS":
            if self.is_real_gpio and self.gpio_lib:
                self.gpio_lib.output(self.pass_pin, self.gpio_lib.HIGH)
                time.sleep(0.15)
                self.gpio_lib.output(self.pass_pin, self.gpio_lib.LOW)
            else:
                logger.debug(f"[MOCK GPIO] PULSE HIGH -> Pass Pin {self.pass_pin} (PASS)")

    def _sound_buzzer(self, freq: int = 1000, duration_sec: float = 0.2) -> None:
        """Audible alert buzzer."""
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(freq, int(duration_sec * 1000))
            else:
                # Terminal bell on Linux/Pi
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass

    def cleanup(self) -> None:
        if self.is_real_gpio and self.gpio_lib:
            try:
                self.gpio_lib.cleanup()
            except Exception:
                pass
