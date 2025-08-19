from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

@dataclass
class Prompt:
    """Container for SAM prompts.

    Attributes
    ----------
    points: Optional[np.ndarray]
        Nx2 array of point coordinates.
    labels: Optional[np.ndarray]
        Array of 0/1 labels corresponding to points.
    box: Optional[Tuple[int,int,int,int]]
        Bounding box x0,y0,x1,y1.
    """
    points: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    box: Optional[Tuple[int, int, int, int]] = None


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
