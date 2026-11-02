import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
import torch.nn.functional as F

from app.services.feature_extractor import extract_features, cosine_similarity


class ReIdentificationEngine:
    """Gallery-based individual tiger re-identification engine using DINOv2 embeddings."""

    def __init__(
        self,
        metadata_path: str = "dataset/metadata.json",
        reference_dir: str = "dataset/reference",
        similarity_threshold: float = 0.72,
    ):
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            # Support execution from backend/ root
            self.metadata_path = Path("../dataset/metadata.json")

        self.reference_dir = Path(reference_dir)
        if not self.reference_dir.exists():
            self.reference_dir = Path("../dataset/reference")

        self.similarity_threshold = similarity_threshold
        self.gallery_embeddings: Dict[str, torch.Tensor] = {}
        self.gallery_info: Dict[str, dict] = {}
        self._load_gallery()

    def _load_gallery(self):
        """Index gallery images from metadata and precompute normalized embeddings."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            data = json.load(f)

        self.gallery_info = data.get("identities", {})

        for tiger_id, info in self.gallery_info.items():
            embeddings_list = []
            for img_name in info.get("gallery_images", []):
                img_file = self.reference_dir / img_name
                if img_file.exists():
                    with Image.open(img_file) as img:
                        emb = extract_features(img)
                        embeddings_list.append(emb)

            if embeddings_list:
                # Store all gallery embeddings for this tiger
                # [N_images, 384]
                self.gallery_embeddings[tiger_id] = torch.stack(embeddings_list)

    def identify(
        self, crop_image: Image.Image, top_k: int = 3
    ) -> Dict:
        """Matches a tiger crop against the gallery embeddings.

        Returns match decision, best identity, confidence score, and candidate ranks.
        """
        query_emb = extract_features(crop_image)  # [384]

        scores_by_tiger: List[Dict] = []

        for tiger_id, emb_matrix in self.gallery_embeddings.items():
            # Cosine similarity with every reference image for this tiger
            sims = torch.mv(emb_matrix, query_emb)  # [N_images]
            best_sim, best_idx = torch.max(sims, dim=0)
            score = float(best_sim.item())
            
            ref_images = self.gallery_info[tiger_id]["gallery_images"]
            best_ref_img = ref_images[best_idx.item()] if best_idx.item() < len(ref_images) else ref_images[0]

            scores_by_tiger.append({
                "tiger_id": tiger_id,
                "name": self.gallery_info[tiger_id].get("name", tiger_id),
                "similarity_score": round(score, 4),
                "reference_image": best_ref_img,
            })

        # Sort descending by similarity
        scores_by_tiger.sort(key=lambda x: x["similarity_score"], reverse=True)

        if not scores_by_tiger:
            return {
                "matched_id": "Unknown",
                "name": "Unknown Tiger",
                "similarity_score": 0.0,
                "is_known": False,
                "reference_image": None,
                "top_candidates": [],
            }

        top_match = scores_by_tiger[0]
        is_known = top_match["similarity_score"] >= self.similarity_threshold

        return {
            "matched_id": top_match["tiger_id"] if is_known else "Unknown",
            "name": top_match["name"] if is_known else "Unknown Tiger",
            "similarity_score": top_match["similarity_score"],
            "is_known": is_known,
            "reference_image": top_match["reference_image"],
            "top_candidates": scores_by_tiger[:top_k],
        }


# Global engine instance
_engine_instance: Optional[ReIdentificationEngine] = None


def get_reid_engine() -> ReIdentificationEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ReIdentificationEngine()
    return _engine_instance
