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

Design notes:
  * Every extractor returns an L2-NORMALISED vector. EmbeddingWorker stores them in a
    FAISS IndexFlatIP, so inner product == cosine similarity only if the vectors are unit
    length. Every path here normalises; none may stop doing so.
  * Image and text vectors must live in the same space for cross-modal search ("find
    frames matching this sentence") to mean anything. SigLIP and CLIP are contrastively
    trained to satisfy that; DINOv2 has no text tower at all, and its _encode_text
    correctly raises rather than returning a vector from an unrelated space.
  * Models are frozen and used in inference mode only: .eval() at construction,
    @torch.no_grad() on every forward.
  * The "keypoints" key is a vestige of the SoIR interface this replaced. It holds a single
    global descriptor, not keypoints.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image


# ── model registry ────────────────────────────────────────────────────────────
# One row per supported backbone. vec_dim must match what the model actually emits — it is
# the width EmbeddingWorker reports, and a mismatch would only surface as a FAISS
# dimensionality error much later. supports_text is documentation; the authoritative check
# is whether the class overrides _encode_text.

MODELS: dict[str, tuple[str, int, bool]] = {
    # name → (hf_model_id, vec_dim, supports_text)
    "siglip_base": ("google/siglip-base-patch16-256",     768,  True),   # default: best size/quality trade-off
    "siglip":      ("google/siglip-so400m-patch14-384",  1152,  True),   # strongest, slowest
    "dinov2_base": ("facebook/dinov2-base",               768,  False),  # image-image only
    "dinov2":      ("facebook/dinov2-large",             1024,  False),
    "clip":        ("openai/clip-vit-base-patch16",       512,  True),   # smallest/fastest
}


def make_extractor(model_name: str, device: str) -> "BaseExtractor":
    """Factory: registry name → a constructed, loaded extractor.

    Dispatch is by name prefix so the two SigLIP (and two DINOv2) sizes share one class —
    they differ only in weights and width, not in how they are called.
    """
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
    # Reachable only if MODELS gains an entry without a matching class above.
    raise ValueError(f"No extractor class mapped for '{model_name}'")


# ── base ──────────────────────────────────────────────────────────────────────

class BaseExtractor:
    """Common interface. Subclasses own their own HF processor/model pair."""

    def __init__(self, hf_id: str, vec_dim: int, device: str):
        self.hf_id   = hf_id
        self.vec_dim = vec_dim
        self.device  = device

    def extract(self, sample: dict, is_query: bool = False) -> dict:
        """Return {"keypoints": Tensor (1, D)} for sample["image"] (PIL.Image).

        `is_query` distinguishes an image being indexed from one being searched with.
        No current extractor treats them differently, but the parameter is part of the
        interface EmbeddingWorker calls through, and asymmetric query/index preprocessing
        is common enough in retrieval to be worth keeping the hook.
        """
        raise NotImplementedError

    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text into the SAME space as extract()'s image vectors.

        The default raises, and that is the correct behaviour for a vision-only backbone —
        EmbeddingWorker.query_text catches NotImplementedError specifically and reports
        "vision-only" rather than returning nonsense from a mismatched space.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support text queries.")


# ── SigLIP ────────────────────────────────────────────────────────────────────

class SigLIPExtractor(BaseExtractor):
    """SigLIP — the project default. Image and text towers share one embedding space."""

    def __init__(self, hf_id: str, vec_dim: int, device: str):
        super().__init__(hf_id, vec_dim, device)
        from transformers import AutoProcessor, AutoModel
        print(f"[Extractor] loading SigLIP: {hf_id} …")
        self._processor = AutoProcessor.from_pretrained(hf_id)
        # .eval() disables dropout/batchnorm updates — these weights are frozen.
        self._model     = AutoModel.from_pretrained(hf_id).to(device).eval()

    @torch.no_grad()
    def extract(self, sample: dict, is_query: bool = False) -> dict:
        img    = sample["image"]
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        # Call vision_model directly rather than the full model: we want the image
        # embedding on its own, not an image-text similarity score.
        out    = self._model.vision_model(**inputs)
        # pooler_output is SigLIP's global image descriptor (attention-pooled, not a CLS
        # token — SigLIP's vision tower has no CLS, unlike DINOv2 below).
        feat   = F.normalize(out.pooler_output, p=2, dim=-1).cpu().float()
        return {"keypoints": feat}

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> torch.Tensor:
        # padding="max_length" is REQUIRED for SigLIP, not a stylistic choice: its text
        # tower was trained with fixed-length padded sequences and produces degraded
        # embeddings under dynamic ("longest") padding.
        inputs = self._processor(text=texts, return_tensors="pt",
                                 padding="max_length", truncation=True).to(self.device)
        out = self._model.text_model(**inputs)
        return F.normalize(out.pooler_output, p=2, dim=-1).cpu().float()


# ── DINOv2 ────────────────────────────────────────────────────────────────────

class DINOv2Extractor(BaseExtractor):
    """DINOv2 — self-supervised, vision-only.

    Strong at pure visual similarity (image-query retrieval), but there is no text tower,
    so `retrieve_memory` with a text query is unavailable under this backbone. It inherits
    BaseExtractor._encode_text, which raises NotImplementedError by design.
    """

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
        # DINOv2 has no pooler; token 0 of the last hidden state is the CLS token, which is
        # the conventional global descriptor. The remaining tokens are per-patch features.
        feat   = out.last_hidden_state[:, 0, :]  # CLS token
        feat   = F.normalize(feat, p=2, dim=-1).cpu().float()
        return {"keypoints": feat}


# ── CLIP ──────────────────────────────────────────────────────────────────────

class CLIPExtractor(BaseExtractor):
    """CLIP — the smallest/fastest cross-modal option. Same structure as SigLIP."""

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
        # Unlike SigLIP, CLIP is fine with dynamic padding to the longest sequence.
        inputs = self._processor(text=texts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
        out = self._model.text_model(**inputs)
        return F.normalize(out.pooler_output, p=2, dim=-1).cpu().float()
