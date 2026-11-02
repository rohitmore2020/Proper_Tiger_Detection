from pathlib import Path
from PIL import Image
import pytest

from app.services.reid_engine import get_reid_engine


def test_reid_known_probe():
    engine = get_reid_engine()
    test_path = Path("dataset/test/000085.jpg")
    if not test_path.exists():
        test_path = Path("../dataset/test/000085.jpg")

    assert test_path.exists(), "Known test probe 000085.jpg must exist"

    with Image.open(test_path) as img:
        res = engine.identify(img)

    assert "matched_id" in res
    assert "similarity_score" in res
    assert res["similarity_score"] > 0.65
    assert len(res["top_candidates"]) > 0


def test_reid_novel_probe_rejection():
    engine = get_reid_engine()
    # 000029.jpg is the isolated single image marked as UNKNOWN in metadata.json
    unknown_path = Path("dataset/test/000029.jpg")
    if not unknown_path.exists():
        unknown_path = Path("../dataset/test/000029.jpg")

    assert unknown_path.exists(), "Unknown test probe 000029.jpg must exist"

    with Image.open(unknown_path) as img:
        res = engine.identify(img)

    # Verify matching structure
    assert "matched_id" in res
    assert "similarity_score" in res
