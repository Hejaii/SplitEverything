from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
from io_utils import save_mask, save_json
from visualize import overlay_masks, build_semantic_map
from prompts_heuristics import auto_segment, PARTS
from postprocess import clean_mask
from sam_wrapper import SamWrapper


def run_pipeline(image_path: Path, out_dir: Path, sam: SamWrapper) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    masks = auto_segment(image, sam)

    processed = {}
    for name in PARTS:
        m = masks.get(name, np.zeros(image.shape[:2], dtype=np.uint8))
        m = clean_mask(m)
        processed[name] = m
        save_mask(out_dir / f"{name}.png", m)

    overlay = overlay_masks(image, processed)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    sem = build_semantic_map(processed, PARTS)
    cv2.imwrite(str(out_dir / "semantics.png"), sem)

    meta = {}
    for idx, name in enumerate(PARTS, start=1):
        m = processed[name]
        area = int(m.sum())
        ys, xs = np.where(m > 0)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bbox = [x0, y0, x1, y1]
        else:
            bbox = [0, 0, 0, 0]
        meta[name] = {"area": area, "bbox": bbox}
    save_json(out_dir / "meta.json", meta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Anime face segmentation demo")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--model-type", default="vit_h")
    parser.add_argument("--sam-checkpoint")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sam = SamWrapper(model_type=args.model_type, checkpoint=args.sam_checkpoint)
    run_pipeline(args.image, args.out, sam)



if __name__ == "__main__":
    main()
