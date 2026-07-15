"""
agent/grounding.py — GroundingDINO wrapper for open-set bounding box proposals.
Lazy-loaded on first call; degrades gracefully if transformers is unavailable.

This backs the agent's `detect` tool. GroundingDINO is an *open-vocabulary* detector:
it takes a free-text query ("water bottle. couch.") rather than a fixed class list, which
is what lets the agent look for whatever the task names without a task-specific model.

The import of torch/transformers is deferred into _load() so that merely importing the
agent package — e.g. in tests, or in a run that never calls `detect` — does not pay the
multi-second CUDA/transformers import cost or hard-fail on a machine without them.
"""
from __future__ import annotations

from typing import Optional

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
        """Record configuration only — no model is loaded until the first detect() call.

        box_threshold  : minimum objectness/confidence for a box to be emitted at all.
        text_threshold : minimum similarity between a box and a query token for that
                         token to be used as the box's label. Lower than box_threshold
                         because label attribution is fuzzier than box confidence.
        device         : "auto" resolves to CUDA when available (decided at load time,
                         not now, so the choice reflects the process that actually runs).
        """
        self._model_id       = model_id
        self._box_threshold  = box_threshold
        self._text_threshold = text_threshold
        self._device_arg     = device
        # Populated by _load(); their None-ness is the "not loaded yet" flag.
        self._processor      = None
        self._model          = None
        self._device: Optional[str] = None

    def _load(self) -> None:
        """Idempotently bring up the processor + model. Safe to call on every detect()."""
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
        # GroundingDINO's text encoder expects phrases terminated by a period; multiple
        # phrases are separated by periods too. We normalise to exactly one trailing '.'
        # so that a query the caller wrote either way ("bottle" / "bottle.") behaves the same.
        text    = query.rstrip(".") + "."
        inputs  = self._processor(
            images=pil_img, text=text, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():                      # inference only — no autograd graph
            outputs = self._model(**inputs)
        # post_process_* rescales boxes from the model's normalised space back to pixels.
        # target_sizes wants (height, width); PIL's .size is (width, height), hence [::-1].
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]                                        # [0] -> the single image in the batch
        # Round to integers/3dp: these values go straight into a text prompt, and
        # sub-pixel precision is noise the VLM would only be distracted by.
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
        # Most-confident first: the agent prompt only shows the top few.
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections
