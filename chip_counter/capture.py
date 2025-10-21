import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from chip_counter.camera.csi import CameraCapture
from chip_counter.hardware.gpio import Button, LED


def timestamped_name(prefix: str = "img", ext: str = ".jpg") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"


def resolve_output_path(output: Path) -> Path:
    if output.exists() and output.is_dir():
        return output / timestamped_name()
    if output.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return output
    # treat as directory if suffix not an image
    return output / timestamped_name()


def run_capture(output: Path, no_wait: bool, button_pin: Optional[int], led_pin: Optional[int]) -> Path:
    button: Optional[Button] = Button(bcm_pin=button_pin) if (button_pin is not None and not no_wait) else None
    led: Optional[LED] = LED(bcm_pin=led_pin) if led_pin is not None else None

    if not no_wait:
        if button is not None:
            print(f"Waiting for button press on BCM pin {button_pin}...")
            button.wait_for_press()
        else:
            input("Press Enter to capture...")

    if led is not None:
        led.on()

    camera = CameraCapture()
    image: np.ndarray = camera.capture_image()

    save_path = resolve_output_path(output)
    camera.save_image_bgr(image, save_path)

    if led is not None:
        led.off()

    print(f"Saved image to {save_path}")
    return save_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a single image for dataset collection")
    parser.add_argument("--output", required=True, type=Path, help="Output file or directory path")
    parser.add_argument("--no-wait", action="store_true", help="Capture immediately without waiting for input/button")
    parser.add_argument("--button-pin", type=int, default=None, help="BCM pin for button (Raspberry Pi)")
    parser.add_argument("--led-pin", type=int, default=None, help="BCM pin for LED (Raspberry Pi)")
    args = parser.parse_args()
    run_capture(args.output, args.no_wait, args.button_pin, args.led_pin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


