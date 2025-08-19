import subprocess
from pathlib import Path
import numpy as np
import cv2

from prompts_heuristics import PARTS


def create_synthetic(path: Path) -> None:
    h, w = 256, 256
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    skin = (200, 180, 150)
    hair = (50, 50, 50)
    mouth = (0, 0, 255)

    # face
    cv2.ellipse(img, (128, 120), (60, 80), 0, 0, 360, skin, -1)
    # hair block
    cv2.rectangle(img, (68, 40), (188, 80), hair, -1)
    # eyes
    cv2.ellipse(img, (100, 110), (15, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (156, 110), (15, 10), 0, 0, 360, (255, 255, 255), -1)
    # mouth
    cv2.ellipse(img, (128, 160), (20, 10), 0, 0, 360, mouth, -1)
    # neck
    cv2.rectangle(img, (108, 200), (148, 240), skin, -1)
    # ears
    cv2.ellipse(img, (68, 120), (12, 20), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (188, 120), (12, 20), 0, 0, 360, skin, -1)

    cv2.imwrite(str(path), img)


def test_cli(tmp_path: Path):
    img_path = tmp_path / "synthetic.png"
    create_synthetic(img_path)
    out_dir = tmp_path / "out"
    cmd = ["python", "segment_anime_face.py", "--image", str(img_path), "--out", str(out_dir), "--auto"]
    subprocess.run(cmd, check=True)

    for part in PARTS:
        mask_path = out_dir / f"{part}.png"
        assert mask_path.exists(), mask_path
        mask = cv2.imread(str(mask_path), 0)
        assert mask.sum() > 0

    hair = cv2.imread(str(out_dir / "hair.png"), 0) > 0
    mouth = cv2.imread(str(out_dir / "mouth.png"), 0) > 0
    overlap = np.logical_and(hair, mouth).sum()
    assert overlap < 0.1 * mouth.sum()
