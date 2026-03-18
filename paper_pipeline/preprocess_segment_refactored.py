# -*- coding: utf-8 -*-
"""Refactored segmentation stage for the upcycled textile pipeline.

What changed from the original script:
- This stage now does one job only: background removal + garment-region crop.
- Random swatch extraction and 10h/10v slicing were moved out of preprocessing.
- Shared scale/threshold values come from ``cloth_pipeline_config.py``.
- File ordering is deterministic and a summary CSV is written for traceability.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from rembg import new_session, remove

from cloth_pipeline_config import (
    ALPHA_THRESHOLD,
    REMBG_MODEL_NAME,
    ensure_dir,
    sorted_image_files,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment garments and save *_nobg.png outputs.")
    parser.add_argument("--input-dir", required=True, help="Folder containing source garment images.")
    parser.add_argument("--output-dir", required=True, help="Folder to save background-removed images.")
    parser.add_argument(
        "--model-name",
        default=REMBG_MODEL_NAME,
        help=f"rembg model name. Default: {REMBG_MODEL_NAME}",
    )
    parser.add_argument(
        "--keep-canvas",
        action="store_true",
        help="Keep original canvas size instead of cropping to the garment bounding box.",
    )
    parser.add_argument(
        "--disable-mask-preview",
        action="store_true",
        help="Skip saving *_mask.png and *_preview.png debug files.",
    )
    return parser.parse_args(argv)


def setup_session(model_name: str):
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("▶ rembg provider: CUDAExecutionProvider")
    else:
        providers = ["CPUExecutionProvider"]
        print("▶ rembg provider: CPUExecutionProvider")
    return new_session(model_name, providers=providers)


def alpha_to_binary_mask(alpha: Image.Image, threshold: int = ALPHA_THRESHOLD) -> np.ndarray:
    alpha_np = np.asarray(alpha, dtype=np.uint8)
    return (alpha_np > threshold).astype(np.uint8) * 255


def bbox_from_mask(mask_bin: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    left = int(xs.min())
    top = int(ys.min())
    right = int(xs.max()) + 1
    bottom = int(ys.max()) + 1
    return left, top, right, bottom


def save_mask_assets(mask_bin: np.ndarray, nobg_image: Image.Image, out_stem: Path) -> None:
    mask_path = out_stem.with_name(f"{out_stem.name}_mask.png")
    preview_path = out_stem.with_name(f"{out_stem.name}_preview.png")

    cv2.imwrite(str(mask_path), mask_bin)

    preview_rgba = np.asarray(nobg_image.convert("RGBA"), dtype=np.uint8).copy()
    preview_bgra = cv2.cvtColor(preview_rgba, cv2.COLOR_RGBA2BGRA)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview_bgra, contours, -1, (0, 255, 0, 255), 2)
    cv2.imwrite(str(preview_path), preview_bgra)


def process_image(
    image_path: Path,
    output_dir: Path,
    session,
    *,
    keep_canvas: bool,
    save_debug_assets: bool,
) -> dict:
    input_image = ImageOps.exif_transpose(Image.open(image_path).convert("RGBA"))
    segmented = remove(input_image, session=session).convert("RGBA")

    alpha = segmented.getchannel("A")
    mask_bin = alpha_to_binary_mask(alpha)
    bbox = bbox_from_mask(mask_bin)
    if bbox is None:
        raise ValueError("No foreground pixels were detected after background removal.")

    if keep_canvas:
        final_image = segmented
        final_mask = mask_bin
        bbox_used = (0, 0, segmented.width, segmented.height)
    else:
        final_image = segmented.crop(bbox)
        left, top, right, bottom = bbox
        final_mask = mask_bin[top:bottom, left:right]
        bbox_used = bbox

    base_name = image_path.stem
    out_stem = output_dir / base_name
    nobg_path = out_stem.with_name(f"{out_stem.name}_nobg.png")
    final_image.save(nobg_path)

    if save_debug_assets:
        save_mask_assets(final_mask, final_image, out_stem)

    left, top, right, bottom = bbox_used
    return {
        "src_image": image_path.name,
        "saved_nobg": nobg_path.name,
        "orig_width_px": input_image.width,
        "orig_height_px": input_image.height,
        "crop_left_px": left,
        "crop_top_px": top,
        "crop_right_px": right,
        "crop_bottom_px": bottom,
        "saved_width_px": final_image.width,
        "saved_height_px": final_image.height,
        "alpha_threshold": ALPHA_THRESHOLD,
        "keep_canvas": int(bool(keep_canvas)),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    image_files = sorted_image_files(input_dir)
    print(f"총 {len(image_files)}개 이미지 처리 시작")
    if not image_files:
        return 0

    session = setup_session(args.model_name)
    summary_path = output_dir / "segmentation_summary.csv"

    fieldnames = [
        "src_image",
        "saved_nobg",
        "orig_width_px",
        "orig_height_px",
        "crop_left_px",
        "crop_top_px",
        "crop_right_px",
        "crop_bottom_px",
        "saved_width_px",
        "saved_height_px",
        "alpha_threshold",
        "keep_canvas",
        "status",
        "error",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, image_path in enumerate(image_files, start=1):
            try:
                row = process_image(
                    image_path,
                    output_dir,
                    session,
                    keep_canvas=args.keep_canvas,
                    save_debug_assets=not args.disable_mask_preview,
                )
                row.update({"status": "ok", "error": ""})
                writer.writerow(row)
                print(f"[{idx}/{len(image_files)}] 완료: {row['saved_nobg']}")
            except Exception as exc:  # pragma: no cover - depends on model/runtime
                writer.writerow(
                    {
                        "src_image": image_path.name,
                        "saved_nobg": "",
                        "orig_width_px": "",
                        "orig_height_px": "",
                        "crop_left_px": "",
                        "crop_top_px": "",
                        "crop_right_px": "",
                        "crop_bottom_px": "",
                        "saved_width_px": "",
                        "saved_height_px": "",
                        "alpha_threshold": ALPHA_THRESHOLD,
                        "keep_canvas": int(bool(args.keep_canvas)),
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"[{idx}/{len(image_files)}] 실패: {image_path.name} | {exc}")

    print(f"✅ segmentation stage finished -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
