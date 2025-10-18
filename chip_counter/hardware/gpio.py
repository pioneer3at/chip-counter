import os
import time
from typing import Optional


def is_raspberry_pi() -> bool:
    # Simple heuristic: presence of /proc/device-tree/model mentioning Raspberry Pi
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read().lower()
        return "raspberry pi" in model
    except Exception:
        return False


class _MockGPIO:
    BCM = "BCM"
    IN = "IN"
    OUT = "OUT"
    PUD_UP = "PUD_UP"

    def setmode(self, mode):
        pass

    def setup(self, pin, direction, pull_up_down=None):
        pass

    def input(self, pin):
        return 0

    def output(self, pin, value):
        pass

    def cleanup(self):
        pass


try:
    import RPi.GPIO as RGPIO  # type: ignore
except Exception:  # Non-Pi environments
    RGPIO = _MockGPIO()  # type: ignore


class Button:
    def __init__(self, bcm_pin: int, pull_up: bool = True):
        self.bcm_pin = bcm_pin
        self.pull_up = pull_up
        RGPIO.setmode(RGPIO.BCM)
        pud = RGPIO.PUD_UP if pull_up else None
        RGPIO.setup(self.bcm_pin, RGPIO.IN, pull_up_down=pud)

    def is_pressed(self) -> bool:
        value = RGPIO.input(self.bcm_pin)
        if self.pull_up:
            return value == 0  # Active-low
        return value == 1

    def wait_for_press(self, debounce_ms: int = 50):
        while True:
            if self.is_pressed():
                time.sleep(debounce_ms / 1000.0)
                if self.is_pressed():
                    return
            time.sleep(0.01)


class LED:
    def __init__(self, bcm_pin: int):
        self.bcm_pin = bcm_pin
        RGPIO.setmode(RGPIO.BCM)
        RGPIO.setup(self.bcm_pin, RGPIO.OUT)
        self.off()

    def on(self):
        RGPIO.output(self.bcm_pin, 1)

    def off(self):
        RGPIO.output(self.bcm_pin, 0)

    def blink(self, times: int = 3, interval_s: float = 0.2):
        for _ in range(times):
            self.on()
            time.sleep(interval_s)
            self.off()
            time.sleep(interval_s)


