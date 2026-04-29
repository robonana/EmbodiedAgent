"""
memory/extractors.py — Self-contained image + text extractors for FAISS retrieval.

Replaces the external SoIR dependency.  All models come from HuggingFace
transformers (already required by this project).

Supported models
----------------
siglip_base  google/siglip-base-patch16-256       768-d  image + text
siglip       google/siglip-so400m-patch14-384    1152-d  image + text
dinov2_base  facebook/dinov2-base                 768-d  image only
dinov2       facebook/dinov2-large               1024-d  image only
clip         openai/clip-vit-base-patch16         512-d  image + text

Each extractor exposes:
    extract(sample, is_query) → {"keypoints": Tensor (1, D)}
    _encode_text(texts)       → Tensor (N, D)   [raises if vision-only]

sample["image"] must be a PIL.Image.Image.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


# ── model registry ────────────────────────────────────────────────────────────

MODELS: dict[str, tuple[str, int, bool]] = {
    # name → (hf_model_id, vec_dim, supports_text)
    "siglip_base": ("google/siglip-base-patch16-256",     768,  True),
    "siglip":      ("google/siglip-so400m-patch14-384",  1152,  True),
    "dinov2_base": ("facebook/dinov2-base",               768,  False),
    "dinov2":      ("facebook/dinov2-large",             1024,  False),
    "clip":        ("openai/clip-vit-base-patch16",       512,  True),
}


def make_extractor(model_name: str, device: str) -> "BaseExtractor":
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from: {list(MODELS)}")
    hf_id, vec_dim, supports_text = MODELS[model_name]
    if model_name.startswith("siglip"):
        return SigLIPExtractor(hf_id, vec_dim, device)
    if model_name.startswith("dinov2"):
        return DINOv2Extractor(hf_id, vec_dim, device)
    if model_name == "clip":
        return CLIPExtractor(hf_id, vec_dim, device)
    raise ValueError(f"No extractor class mapped for '{model_name}'")


# ── base ──────────────────────────────────────────────────────────────────────

class BaseExtractor:
    def __init__(self, hf_id: str, vec_dim: int, device: str):
        self.hf_id   = hf_id
        self.vec_dim = vec_dim
        self.device  = device

    def extract(self, sample: dict, is_query: bool = False) -> dict:
        """Return {"keypoints": Tensor (1, D)} for sample["image"] (PIL.Image)."""
        raise NotImplementedError

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support text queries.")


# ── SigLIP ────────────────────────────────────────────────────────────────────

class SigLIPExtractor(BaseExtractor):
    def __init__(self, hf_id: str, vec_dim: int, device: str):
        super().__init__(hf_id, vec_dim, device)
        from transformers import AutoProcessor, AutoModel
        print(f"[Extractor] loading SigLIP: {hf_id} …")
        self._processor = AutoProcessor.from_pretrained(hf_id)
        self._model     = AutoModel.from_pretrained(hf_id).to(device).eval()

    @torch.no_grad()
    def extract(self, sample: dict, is_query: bool = False) -> dict:
        img    = sample["image"]
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        out    = self._model.vision_model(**inputs)
        feat   = F.normalize(out.pooler_output, p=2, dim=-1).cpu().float()
        return {"keypoints": feat}

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self._processor(text=texts, return_tensors="pt",
                                 padding="max_length", truncation=True).to(self.device)
        out = self._model.text_model(**inputs)
        return F.normalize(out.pooler_output, p=2, dim=-1).cpu().float()


# ── DINOv2 ────────────────────────────────────────────────────────────────────

class DINOv2Extractor(BaseExtractor):
    def __init__(self, hf_id: str, vec_dim: int, device: str):
        super().__init__(hf_id, vec_dim, device)
        from transformers import AutoImageProcessor, AutoModel
        print(f"[Extractor] loading DINOv2: {hf_id} …")
        self._processor = AutoImageProcessor.from_pretrained(hf_id)
        self._model     = AutoModel.from_pretrained(hf_id).to(device).eval()

    @torch.no_grad()
    def extract(self, sample: dict, is_query: bool = False) -> dict:
        img    = sample["image"]
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        out    = self._model(**inputs)
        feat   = out.last_hidden_state[:, 0, :]  # CLS token
        feat   = F.normalize(feat, p=2, dim=-1).cpu().float()
        return {"keypoints": feat}


# ── CLIP ──────────────────────────────────────────────────────────────────────

class CLIPExtractor(BaseExtractor):
    def __init__(self, hf_id: str, vec_dim: int, device: str):
        super().__init__(hf_id, vec_dim, device)
        from transformers import CLIPProcessor, CLIPModel
        print(f"[Extractor] loading CLIP: {hf_id} …")
        self._processor = CLIPProcessor.from_pretrained(hf_id)
        self._model     = CLIPModel.from_pretrained(hf_id).to(device).eval()

    @torch.no_grad()
    def extract(self, sample: dict, is_query: bool = False) -> dict:
        img    = sample["image"]
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        out    = self._model.vision_model(**inputs)
        feat   = F.normalize(out.pooler_output, p=2, dim=-1).cpu().float()
        return {"keypoints": feat}

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        inputs = self._processor(text=texts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
        out = self._model.text_model(**inputs)
        return F.normalize(out.pooler_output, p=2, dim=-1).cpu().float()
