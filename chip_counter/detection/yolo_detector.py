from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # xyxy
    class_id: int
    class_name: str
    confidence: float
    count: int


class YOLODetector:
    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int | None = None,
    ):
        from ultralytics import YOLO  # lazy import

        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        # Class names are derived from the model metadata
        self.class_names = self.model.names

    def predict(self, image_rgb: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=image_rgb,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz if self.imgsz is not None else None,
        )
        detections: List[Detection] = []
        for r in results:
            boxes = r.boxes  # type: ignore
            if boxes is None:
                continue
            for b in boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                xyxy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = [float(v) for v in xyxy]
                name = self.class_names.get(cls_id, str(cls_id))
                # Expect class names like "count_1", "count_2", ... or numeric indexes
                count = self._class_to_count(cls_id, name)
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=name,
                        confidence=conf,
                        count=count,
                    )
                )
        return detections

    def predict_total_count(self, image_rgb: np.ndarray) -> Tuple[int, List[Dict]]:
        dets = self.predict(image_rgb)
        total = sum(d.count for d in dets)
        # Return a simple serializable dict list for CLI printing
        det_list = [
            {
                "bbox": d.bbox,
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "count": d.count,
            }
            for d in dets
        ]
        return total, det_list

    @staticmethod
    def _class_to_count(class_id: int, class_name: str) -> int:
        # Prefer parsing the class name like "count_7". Fallback to (class_id + 1).
        if isinstance(class_name, str) and class_name.startswith("count_"):
            try:
                return int(class_name.split("_")[1])
            except Exception:
                pass
        # If names are just provided as {0: '1', 1: '2', ...}
        try:
            return int(class_name)
        except Exception:
            return class_id + 1


