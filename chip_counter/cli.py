import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from chip_counter.hardware.gpio import Button, LED
from chip_counter.camera.csi import CameraCapture
from chip_counter.detection.yolo_detector import YOLODetector


def run(weights: str, button_pin: Optional[int], led_pin: Optional[int], save_image: Optional[str]) -> int:
    button: Optional[Button] = Button(bcm_pin=button_pin) if button_pin is not None else None
    led: Optional[LED] = LED(bcm_pin=led_pin) if led_pin is not None else None

    if button is not None:
        print(f"Waiting for button press on BCM pin {button_pin}...")
        button.wait_for_press()
    else:
        input("Press Enter to capture...")

    if led is not None:
        led.on()

    camera = CameraCapture()
    image: np.ndarray = camera.capture_image()

    if save_image:
        camera.save_image_bgr(image, Path(save_image))

    detector = YOLODetector(weights_path=weights)
    total_count, detections = detector.predict_total_count(image)

    if led is not None:
        led.off()

    print("Detections:")
    for det in detections:
        print(det)
    print(f"Total chips: {total_count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chip Counter")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 weights .pt")
    parser.add_argument("--button-pin", type=int, default=None, help="BCM pin for button (Raspberry Pi)")
    parser.add_argument("--led-pin", type=int, default=None, help="BCM pin for LED (Raspberry Pi)")
    parser.add_argument("--save-image", type=str, default=None, help="Optional path to save captured image")
    args = parser.parse_args()
    return run(args.weights, args.button_pin, args.led_pin, args.save_image)


