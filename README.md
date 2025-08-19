# SplitEverything

Anime face semantic part segmentation using Meta's Segment Anything Model (SAM)
and text-driven GroundingDINO detections.

## Setup

Install dependencies (CPU-only PyTorch wheels are used) and download required assets:
```bash
python -m pip install -r requirements.txt
mkdir -p weights assets
curl -L -o weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
curl -L -o assets/lbpcascade_animeface.xml \
  https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml
# optional: GroundingDINO weights for text-driven boxes
# curl -L -o weights/groundingdino_swint_ogc.pth https://example.com/groundingdino_swint_ogc.pth
# curl -L -o weights/GroundingDINO_SwinT_OGC.py https://example.com/GroundingDINO_SwinT_OGC.py
```

## Usage

Segment an image into neck, eyes, mouth, hair, and ears.  With both SAM and
GroundingDINO checkpoints provided the pipeline runs a text prompt → DINO → box
→ SAM refinement.  When either model is missing the script falls back to simple
color heuristics (sufficient for the unit tests):
```bash
python segment_anime_face.py \
    --image path/to/input.jpg \
    --out output_dir \
    --auto \
    --model-type vit_h \
    --sam-checkpoint weights/sam_vit_h_4b8939.pth \
    --dino-config weights/GroundingDINO_SwinT_OGC.py \
    --dino-checkpoint weights/groundingdino_swint_ogc.pth
```
Outputs include binary masks (`neck.png`, `eyes.png`, `mouth.png`, `hair.png`, `ears.png`),
a semantic label map (`semantics.png`), an overlay visualization (`overlay.png`),
and a `meta.json` file containing mask statistics and prompts.

## Testing

Run the end-to-end test on a synthetic anime face:
```bash
pytest -q
```
