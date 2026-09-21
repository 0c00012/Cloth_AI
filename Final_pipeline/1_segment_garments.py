# -*- coding: utf-8 -*-
"""
Step 1 - Garment segmentation (paper 3.3).

* Pretrained U^2-Net through `rembg` (no fine-tuning) removes the background and
  returns an RGBA image whose alpha channel carries the garment foreground.
* The RGBA result is cropped to the alpha bounding box and saved as <stem>_nobg.png.
* A binary garment mask (alpha > ALPHA_THRESHOLD -> 255, else 0) is saved as
  <stem>_mask.png. The same rule is used by every downstream step.

Inputs : data/input_images/*.jpg|png   (3,755 x 2,628 px, 100 cm == 3,755 px)
Outputs: work/01_segmented/<stem>_nobg.png, <stem>_mask.png, <stem>_preview.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline_config import ALPHA_THRESHOLD, INPUT_IMAGE_DIR, REMBG_MODEL, SEGMENT_DIR

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def make_session():
    import onnxruntime as ort
    from rembg import new_session

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print(f"rembg model={REMBG_MODEL} providers={providers}")
    return new_session(REMBG_MODEL, providers=providers)


def segment_one(path: Path, session, out_dir: Path) -> dict:
    from rembg import remove

    rgba = remove(Image.open(path).convert("RGBA"), session=session)
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()                      # tight box of alpha > 0
    if bbox is None:
        raise RuntimeError(f"no foreground detected: {path.name}")
    rgba = rgba.crop(bbox)
    stem = path.stem
    rgba.save(out_dir / f"{stem}_nobg.png")

    a = np.array(rgba.getchannel("A"))
    mask = (a > ALPHA_THRESHOLD).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_dir / f"{stem}_mask.png")

    preview = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    preview.alpha_composite(rgba)
    preview.convert("RGB").save(out_dir / f"{stem}_preview.png")
    return {
        "source": path.name, "nobg": f"{stem}_nobg.png", "mask": f"{stem}_mask.png",
        "crop_left": bbox[0], "crop_top": bbox[1], "crop_right": bbox[2], "crop_bottom": bbox[3],
        "width": rgba.width, "height": rgba.height,
        "foreground_px": int((a > ALPHA_THRESHOLD).sum()), "alpha_threshold": ALPHA_THRESHOLD,
    }


def main(argv=None) -> int:
    in_dir = Path(argv[0]) if argv else INPUT_IMAGE_DIR
    out_dir = Path(argv[1]) if argv and len(argv) > 1 else SEGMENT_DIR
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"no input images in {in_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()
    rows = []
    for i, f in enumerate(files, 1):
        row = segment_one(f, session, out_dir)
        rows.append(row)
        print(f"[{i}/{len(files)}] {f.name}: {row['width']}x{row['height']} px, foreground {row['foreground_px']} px")
    with (out_dir / "segmentation_summary.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
