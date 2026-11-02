from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from PIL import Image
from ultralytics import YOLO


# Quadruped / animal COCO classes where tigers appear in generic YOLO models
ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


class TigerDetector:
    def __init__(self, model_weight: str = "yolov8n.pt", conf_thresh: float = 0.25):
        self.conf_thresh = conf_thresh
        self.model = YOLO(model_weight)

    def detect_and_crop(
        self, image: Image.Image, padding_pct: float = 0.05
    ) -> Tuple[bool, Optional[List[float]], float, Image.Image]:
        """Detects the tiger/animal in the image and returns a padded crop.

        Returns:
            (detected: bool, bbox: [x1, y1, x2, y2], confidence: float, crop: PIL.Image)
        """
        w, h = image.size
        # Run inference
        results = self.model(image, conf=self.conf_thresh, verbose=False)[0]
        boxes = results.boxes

        best_box = None
        best_conf = 0.0

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())

                # Prioritize large animal / predator classes
                if cls_id in ANIMAL_CLASSES:
                    if conf > best_conf:
                        best_conf = conf
                        best_box = b.xyxy[0].tolist()

            # If no animal class hit but another object detected with high confidence
            if best_box is None and len(boxes) > 0:
                best_box = boxes[0].xyxy[0].tolist()
                best_conf = float(boxes[0].conf[0].item())

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            # Add padding
            pad_w = (x2 - x1) * padding_pct
            pad_h = (y2 - y1) * padding_pct

            x1_pad = max(0, int(x1 - pad_w))
            y1_pad = max(0, int(y1 - pad_h))
            x2_pad = min(w, int(x2 + pad_w))
            y2_pad = min(h, int(y2 + pad_h))

            crop = image.crop((x1_pad, y1_pad, x2_pad, y2_pad))
            bbox = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            return True, bbox, round(best_conf, 3), crop

        # Fallback: if no box is identified, treat the entire image as the crop
        return False, [0.0, 0.0, float(w), float(h)], 0.0, image.copy()


# Global detector instance
_detector_instance: Optional[TigerDetector] = None


def get_detector() -> TigerDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = TigerDetector()
    return _detector_instance
