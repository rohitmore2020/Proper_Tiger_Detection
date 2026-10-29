
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-small"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()


@torch.inference_mode()
def extract_features(image: Image.Image) -> torch.Tensor:
    """Return a normalized feature vector for an image."""
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :]
    features = F.normalize(features, p=2, dim=1)

    return features.squeeze(0)


def cosine_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    """Compare two normalized image feature vectors."""
    return F.cosine_similarity(first, second, dim=0).item()
