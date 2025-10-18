from pathlib import Path
from typing import Optional

import numpy as np


class CameraCapture:
    def __init__(self, prefer_picamera2: bool = True, device_index: int = 0):
        self.prefer_picamera2 = prefer_picamera2
        self.device_index = device_index
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        if self.prefer_picamera2:
            try:
                from picamera2 import Picamera2  # type: ignore

                self._backend = ("picamera2", Picamera2())
                return
            except Exception:
                pass
        # fallback to OpenCV
        import cv2  # type: ignore

        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera device")
        self._backend = ("opencv", cap)

    def capture_image(self) -> np.ndarray:
        backend, obj = self._backend
        if backend == "picamera2":
            picam2 = obj
            picam2.configure(picam2.create_still_configuration())
            picam2.start()
            frame = picam2.capture_array()
            picam2.stop()
            # Picamera2 returns RGB
            return frame.copy()
        else:
            import cv2  # type: ignore

            cap = obj
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError("Failed to read from camera")
            # Convert to RGB for detector
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return frame_rgb

    @staticmethod
    def save_image_bgr(image_rgb: np.ndarray, out_path: Path) -> None:
        import cv2  # type: ignore

        out_path.parent.mkdir(parents=True, exist_ok=True)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), image_bgr)


