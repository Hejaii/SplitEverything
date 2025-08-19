from __future__ import annotations
from typing import Optional
import numpy as np
import cv2

try:
    from segment_anything import SamPredictor, sam_model_registry
    _SAM_AVAILABLE = True
except Exception:  # pragma: no cover - SAM may not be installed in tests
    SamPredictor = None  # type: ignore
    sam_model_registry = {}
    _SAM_AVAILABLE = False


class SamWrapper:
    """Minimal wrapper around Meta SAM with graceful fallback.

    When SAM is not available or checkpoint cannot be loaded, the wrapper
    falls back to returning simple masks constructed from prompt boxes or
    positive points.  This makes the rest of the pipeline testable without
    heavy dependencies while keeping the API similar to SAM's predictor.
    """

    def __init__(self, model_type: str = "vit_h", checkpoint: Optional[str] = None):
        self.use_sam = False
        if _SAM_AVAILABLE and checkpoint is not None:
            try:
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                self.predictor = SamPredictor(sam)
                self.use_sam = True
            except Exception:
                self.predictor = None
        else:
            self.predictor = None
        self.image: Optional[np.ndarray] = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image
        if self.use_sam:
            self.predictor.set_image(image)

    def predict(self, prompt) -> np.ndarray:
        if self.use_sam and self.predictor is not None:
            points = prompt.points
            labels = prompt.labels
            box = prompt.box
            masks, _, _ = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=False,
            )
            return masks[0]
        # fallback: build mask from box / positive points
        h, w, _ = self.image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        if prompt.box is not None:
            x0, y0, x1, y1 = map(int, prompt.box)
            mask[y0:y1, x0:x1] = 1
        if prompt.points is not None and prompt.labels is not None:
            for (x, y), label in zip(prompt.points, prompt.labels):
                if label == 1:
                    cv2.circle(mask, (int(x), int(y)), 5, 1, -1)
        return mask
