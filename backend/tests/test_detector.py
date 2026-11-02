from pathlib import Path
from PIL import Image
import pytest

from app.services.detector import get_detector


def test_detector_on_reference_image():
    detector = get_detector()
    img_path = Path("dataset/reference/000001.jpg")
    if not img_path.exists():
        img_path = Path("../dataset/reference/000001.jpg")

    assert img_path.exists(), "Sample test image 000001.jpg must exist"

    with Image.open(img_path) as img:
        detected, bbox, conf, crop = detector.detect_and_crop(img)

    assert detected is True, "Tiger should be detected in 000001.jpg"
    assert bbox is not None and len(bbox) == 4, "Bounding box must have 4 coordinates"
    assert conf > 0.25, f"Confidence score should exceed 0.25, got {conf}"
    assert crop.width > 50 and crop.height > 50, "Crop should be non-trivial"
