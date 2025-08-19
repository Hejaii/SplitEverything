from __future__ import annotations
from typing import Dict
import numpy as np
import cv2

PART_COLORS = {
    'neck': (255, 0, 255),
    'eyes': (0, 255, 0),
    'mouth': (0, 0, 255),
    'hair': (255, 0, 0),
    'ears': (0, 255, 255),
}


def overlay_masks(image: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
    overlay = image.copy()
    for part, mask in masks.items():
        color = PART_COLORS.get(part, (255, 255, 255))
        overlay[mask > 0] = (
            0.5 * overlay[mask > 0] + 0.5 * np.array(color)
        ).astype(np.uint8)
    return overlay


def build_semantic_map(masks: Dict[str, np.ndarray], order: list[str]) -> np.ndarray:
    h, w = next(iter(masks.values())).shape
    sem = np.zeros((h, w), dtype=np.uint8)
    for idx, name in enumerate(order, start=1):
        m = masks.get(name)
        if m is not None:
            sem[m > 0] = idx
    return sem
