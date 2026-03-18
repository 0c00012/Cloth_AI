# -*- coding: utf-8 -*-
"""Optional utility: split swatches into 10 horizontal and 10 vertical slices.

This was extracted from preprocessing so the main pipeline stays modular.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageOps

from cloth_pipeline_config import SLICE_PARTS, cumulative_starts, ensure_dir, sorted_image_files, split_sizes

SKIP_SUFFIXES = ("_packing_vis", "_occupied_mask", "_mask", "_preview")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split swatch images into 10h / 10v strips.")
    parser.add_argument("--input-dir", required=True, help="Input root containing swatch PNG files.")
    parser.add_argument("--output-dir", required=True, help="Output root for sliced swatches.")
    parser.add_argument(
        "--parts",
        type=int,
        default=SLICE_PARTS,
        help=f"Number of horizontal / vertical parts. Default: {SLICE_PARTS}",
    )
    return parser.parse_args(argv)


def should_process(path: Path) -> bool:
    stem = path.stem
    if any(stem.endswith(suffix) for suffix in SKIP_SUFFIXES):
        return False
    return True


def split_one_image(image_path: Path, output_root: Path, input_root: Path, parts: int) -> dict:
    image = ImageOps.exif_transpose(Image.open(image_path).convert("RGBA"))
    width, height = image.size

    relative_parent = image_path.parent.relative_to(input_root)
    swatch_dir = ensure_dir(output_root / relative_parent / image_path.stem)

    h_sizes = split_sizes(height, parts)
    h_starts = cumulative_starts(h_sizes)
    for idx, (y, hh) in enumerate(zip(h_starts, h_sizes)):
        band = image.crop((0, y, width, y + hh))
        band.save(swatch_dir / f"{image_path.stem}_H_{idx:02d}.png")

    w_sizes = split_sizes(width, parts)
    w_starts = cumulative_starts(w_sizes)
    for idx, (x, ww) in enumerate(zip(w_starts, w_sizes)):
        band = image.crop((x, 0, x + ww, height))
        band.save(swatch_dir / f"{image_path.stem}_V_{idx:02d}.png")

    return {
        "src_image": str(image_path.relative_to(input_root)),
        "output_dir": str(swatch_dir.relative_to(output_root)),
        "width_px": width,
        "height_px": height,
        "parts": parts,
        "num_outputs": parts * 2,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_root = Path(args.input_dir)
    output_root = ensure_dir(args.output_dir)

    if not input_root.exists():
        raise SystemExit(f"Input directory does not exist: {input_root}")

    image_files = [path for path in sorted_image_files(input_root, recursive=True) if should_process(path)]
    summary_path = output_root / "slice_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["src_image", "output_dir", "width_px", "height_px", "parts", "num_outputs", "status", "error"],
        )
        writer.writeheader()

        for idx, image_path in enumerate(image_files, start=1):
            try:
                row = split_one_image(image_path, output_root, input_root, args.parts)
                row.update({"status": "ok", "error": ""})
                writer.writerow(row)
                print(f"[{idx}/{len(image_files)}] 완료: {image_path.name}")
            except Exception as exc:
                writer.writerow(
                    {
                        "src_image": str(image_path.relative_to(input_root)),
                        "output_dir": "",
                        "width_px": "",
                        "height_px": "",
                        "parts": args.parts,
                        "num_outputs": 0,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"[{idx}/{len(image_files)}] 실패: {image_path.name} | {exc}")

    print(f"✅ slicing finished -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
