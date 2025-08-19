from __future__ import annotations
from typing import Dict, Tuple, TYPE_CHECKING
import numpy as np
import cv2
from io_utils import Prompt
from postprocess import clean_mask

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from sam_wrapper import SamWrapper



PARTS = ["neck", "eyes", "mouth", "hair", "ears"]


def _heuristic_masks(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Simple color based segmentation for synthetic anime faces."""

    h, w, _ = image.shape

    # Thresholds for different colors (BGR)
    skin = cv2.inRange(image, (170, 150, 120), (255, 220, 200))
    eyes = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
    mouth = cv2.inRange(image, (0, 0, 200), (80, 80, 255))
    hair = cv2.inRange(image, (0, 0, 0), (80, 80, 80))

    skin = clean_mask(skin)
    eyes = clean_mask(eyes)
    mouth = clean_mask(mouth)
    hair = clean_mask(hair)


    coords = cv2.findNonZero(skin)
    x, y, w_box, h_box = cv2.boundingRect(coords)

    neck = np.zeros_like(skin)
    y0 = y + int(0.6 * h_box)
    y1 = min(h, y + h_box + int(0.3 * h_box))
    x0 = x + int(0.2 * w_box)
    x1 = x + int(0.8 * w_box)
    neck[y0:y1, x0:x1] = 255
    neck = neck & skin
    neck = clean_mask(neck)

    eyes_mask = eyes.copy()
    eyes_mask[: y + int(0.2 * h_box), :] = 0
    eyes_mask[y + int(0.6 * h_box) :, :] = 0

    mouth_mask = mouth.copy()
    mouth_mask[: y + int(0.55 * h_box), :] = 0

    hair_mask = hair.copy()
    hair_mask[y + int(0.2 * h_box) :, :] = 0

    ears_mask = skin.copy()
    ears_mask[: y + int(0.2 * h_box), :] = 0
    ears_mask[y + int(0.8 * h_box) :, :] = 0
    ears_mask[:, x + int(0.15 * w_box) : x + int(0.85 * w_box)] = 0
    ears_mask = clean_mask(ears_mask)

    return {
        "neck": neck,
        "eyes": eyes_mask,
        "mouth": mouth_mask,
        "hair": hair_mask,
        "ears": ears_mask,
    }


def _generate_prompts(masks: Dict[str, np.ndarray]) -> Dict[str, Prompt]:
    prompts: Dict[str, Prompt] = {}
    for part, m in masks.items():
        coords = cv2.findNonZero(m)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            prompts[part] = Prompt(box=(x, y, x + w, y + h))
        else:
            prompts[part] = Prompt()
    return prompts


def auto_segment(image: np.ndarray, sam: "SamWrapper") -> Dict[str, np.ndarray]:
    """Segment image into parts using SAM when available.

    If SAM is not available the function falls back to deterministic color
    based masks suitable for the synthetic unit test image.
    """
    masks = _heuristic_masks(image)
    prompts = _generate_prompts(masks)
    result: Dict[str, np.ndarray] = {}

    sam.set_image(image)
    for part in PARTS:
        if sam.use_sam:
            result[part] = sam.predict(prompts[part]).astype(np.uint8) * 255
        else:
            result[part] = masks[part]
    return result
