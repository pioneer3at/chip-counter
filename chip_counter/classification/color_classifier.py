from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ColorPrediction:
    class_id: int
    class_name: str
    confidence: float


class YOLOColorClassifier:
    def __init__(self, weights_path: str):
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(weights_path)  # e.g., yolov8n-cls.pt fine-tuned
        self.class_names = self.model.names

    def predict(self, crop_rgb: np.ndarray) -> ColorPrediction:
        # Ultralytics classify expects RGB np.ndarray as source
        results = self.model.predict(source=crop_rgb, verbose=False)
        r = results[0]
        # r.probs contains classification probs
        probs = r.probs  # type: ignore
        class_id = int(probs.top1)
        conf = float(probs.top1conf)
        class_name = self.class_names.get(class_id, str(class_id))
        return ColorPrediction(class_id=class_id, class_name=class_name, confidence=conf)


