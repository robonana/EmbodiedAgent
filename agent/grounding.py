"""
agent/grounding.py — GroundingDINO wrapper for open-set bounding box proposals.
Lazy-loaded on first call; degrades gracefully if transformers is unavailable.
"""
from __future__ import annotations

import numpy as np


class GroundingDINODetector:
    """Open-set bounding box proposals via GroundingDINO (transformers backend)."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        device: str = "auto",
    ):
        self._model_id       = model_id
        self._box_threshold  = box_threshold
        self._text_threshold = text_threshold
        self._device_arg     = device
        self._processor      = None
        self._model          = None
        self._device: str | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = self._device_arg
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device    = device
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model     = (AutoModelForZeroShotObjectDetection
                           .from_pretrained(self._model_id).to(device))
        print(f"[GroundingDINO] loaded {self._model_id} on {device}")

    def detect(
        self,
        image: np.ndarray,  # HxWx3 uint8
        query: str,
    ) -> list[dict]:
        """
        Returns list of {"bbox": [x1,y1,x2,y2], "label": str, "score": float}
        in absolute pixel coordinates, sorted by score descending.
        """
        self._load()
        import torch
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        text    = query.rstrip(".") + "."
        inputs  = self._processor(
            images=pil_img, text=text, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]
        detections = [
            {
                "bbox":  [round(float(v)) for v in box.tolist()],
                "label": label,
                "score": round(float(score), 3),
            }
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            )
        ]
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections
