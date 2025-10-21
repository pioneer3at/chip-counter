import argparse
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import yaml

from chip_counter.hardware.gpio import Button, LED
from chip_counter.camera.csi import CameraCapture
from chip_counter.detection.yolo_detector import YOLODetector


def run(
    weights: str,
    button_pin: Optional[int],
    led_pin: Optional[int],
    save_image: Optional[str],
    denom_yaml: Optional[str],
    conf: float,
    iou: float,
) -> int:
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

    detector = YOLODetector(weights_path=weights, conf_threshold=conf, iou_threshold=iou)
    total_count, detections = detector.predict_total_count(image)

    if led is not None:
        led.off()

    print("Detections:")
    for det in detections:
        print(det)

    print(f"Total chips: {total_count}")

    # Optional: denomination mapping to compute monetary total
    if denom_yaml:
        try:
            with open(denom_yaml, "r", encoding="utf-8") as f:
                denom_map: Dict[str, int] = yaml.safe_load(f) or {}
            total_value = 0
            for det in detections:
                cls = det.get("class_name", "")
                total_value += int(denom_map.get(cls, 0))
            print(f"Total value: {total_value}")
        except Exception as e:
            print(f"Warning: failed to load denominations from {denom_yaml}: {e}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chip Counter")
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLOv8 weights .pt")
    parser.add_argument("--button-pin", type=int, default=None, help="BCM pin for button (Raspberry Pi)")
    parser.add_argument("--led-pin", type=int, default=None, help="BCM pin for LED (Raspberry Pi)")
    parser.add_argument("--save-image", type=str, default=None, help="Optional path to save captured image")
    parser.add_argument("--denom-yaml", type=str, default=None, help="Path to YAML mapping class_name -> value")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold (default: 0.45)")
    args = parser.parse_args()
    return run(args.weights, args.button_pin, args.led_pin, args.save_image, args.denom_yaml, args.conf, args.iou)


