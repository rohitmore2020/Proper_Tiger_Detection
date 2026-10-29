
from pathlib import Path

from PIL import Image

from app.services.feature_extractor import (
    cosine_similarity,
    extract_features,
)


def main():
    paths = sorted(Path("../dataset/reference").glob("*.jpg"))

    if len(paths) < 2:
        raise RuntimeError("Need at least two reference images.")

    with Image.open(paths[0]) as image:
        first = extract_features(image)

    with Image.open(paths[1]) as image:
        second = extract_features(image)

    print("Image 1:", paths[0].name)
    print("Image 2:", paths[1].name)
    print("Feature dimensions:", first.shape)
    print("Cosine similarity:", cosine_similarity(first, second))


if __name__ == "__main__":
    main()
