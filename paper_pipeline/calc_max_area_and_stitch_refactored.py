# -*- coding: utf-8 -*-
"""Refactored greedy swatch packing.

Key improvements over the original implementation:
- Uses exact square-in-mask validation via an integral image.
- Fixes off-by-one scan coverage at the right/bottom edges.
- Supports multiple scan offsets and keeps the best greedy packing result.
- Saves deterministic metadata for each swatch and each source image.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from cloth_pipeline_config import (
    ALPHA_THRESHOLD,
    PACK_OFFSET_MODE,
    PACK_SCAN_STEP_PX,
    SWATCH_CM,
    SWATCH_PX,
    ensure_dir,
    px_to_cm,
    sorted_image_files,
)


@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    size_px: int

    @property
    def right(self) -> int:
        return self.x + self.size_px

    @property
    def bottom(self) -> int:
        return self.y + self.size_px


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greedy packing of 10cm x 10cm swatches.")
    parser.add_argument("--input-dir", required=True, help="Folder containing *_nobg.png images.")
    parser.add_argument("--output-dir", required=True, help="Folder to save packed swatches.")
    parser.add_argument(
        "--scan-step-px",
        type=int,
        default=PACK_SCAN_STEP_PX,
        help=f"Grid scan step in pixels. Default: {PACK_SCAN_STEP_PX}",
    )
    parser.add_argument(
        "--offset-mode",
        choices=["single", "coarse", "full"],
        default=PACK_OFFSET_MODE,
        help="Offset search mode for the greedy scan.",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=ALPHA_THRESHOLD,
        help=f"Binary mask threshold for the alpha channel. Default: {ALPHA_THRESHOLD}",
    )
    return parser.parse_args(argv)


def rect_sum(integral: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    x2 = x + w
    y2 = y + h
    return int(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])


def build_integral(binary_mask: np.ndarray) -> np.ndarray:
    integral = np.zeros((binary_mask.shape[0] + 1, binary_mask.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = binary_mask.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    return integral


def generate_offsets(step: int, mode: str) -> List[tuple[int, int]]:
    if step <= 0:
        raise ValueError("scan step must be positive")

    if mode == "single":
        offsets = [(0, 0)]
    elif mode == "coarse":
        raw = sorted({0, step // 2})
        offsets = [(ox, oy) for oy in raw for ox in raw]
    else:  # full
        offsets = [(ox, oy) for oy in range(step) for ox in range(step)]

    return offsets


def scan_positions(limit: int, size: int, step: int, offset: int) -> List[int]:
    if limit < size:
        return []

    max_start = limit - size
    offset = max(0, min(offset, max_start))

    forward = list(range(offset, max_start + 1, step))
    backward_anchor = max(0, max_start - offset)
    backward = list(range(backward_anchor, -1, -step))

    positions = set(forward)
    positions.update(backward)
    positions.add(max_start)
    return sorted(p for p in positions if 0 <= p <= max_start)


def greedy_pack(
    fabric_mask: np.ndarray,
    *,
    size_px: int,
    scan_step_px: int,
    offset_mode: str,
) -> tuple[List[Placement], tuple[int, int], np.ndarray]:
    h, w = fabric_mask.shape
    window_area = size_px * size_px
    fabric_integral = build_integral(fabric_mask)

    best_placements: List[Placement] = []
    best_offset = (0, 0)
    best_occupied = np.zeros_like(fabric_mask, dtype=bool)

    for offset_x, offset_y in generate_offsets(scan_step_px, offset_mode):
        xs = scan_positions(w, size_px, scan_step_px, offset_x)
        ys = scan_positions(h, size_px, scan_step_px, offset_y)
        occupied = np.zeros_like(fabric_mask, dtype=bool)
        placements: List[Placement] = []

        for y in ys:
            for x in xs:
                if rect_sum(fabric_integral, x, y, size_px, size_px) != window_area:
                    continue
                if occupied[y : y + size_px, x : x + size_px].any():
                    continue

                occupied[y : y + size_px, x : x + size_px] = True
                placements.append(Placement(x=x, y=y, size_px=size_px))

        if len(placements) > len(best_placements):
            best_placements = placements
            best_offset = (offset_x, offset_y)
            best_occupied = occupied.copy()

    return best_placements, best_offset, best_occupied


def load_rgba_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    if image.ndim != 3 or image.shape[2] < 4:
        raise ValueError("Expected an RGBA image with an alpha channel.")
    return image


def save_visualization(image_bgra: np.ndarray, placements: Iterable[Placement], out_path: Path) -> None:
    vis = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
    for placement in placements:
        cv2.rectangle(
            vis,
            (placement.x, placement.y),
            (placement.right, placement.bottom),
            (0, 0, 255),
            2,
        )
    cv2.imwrite(str(out_path), vis)


def pack_swatches_for_image(
    image_path: Path,
    output_dir: Path,
    *,
    scan_step_px: int,
    offset_mode: str,
    alpha_threshold: int,
) -> dict:
    image = load_rgba_image(image_path)
    h, w = image.shape[:2]
    alpha = image[:, :, 3]
    fabric_mask = (alpha > alpha_threshold).astype(np.uint8)

    if fabric_mask.sum() == 0:
        raise ValueError("Foreground mask is empty after thresholding.")
    if h < SWATCH_PX or w < SWATCH_PX:
        raise ValueError(
            f"Image is smaller than one swatch ({SWATCH_PX}px). Actual size: {w}x{h}px"
        )

    placements, best_offset, occupied = greedy_pack(
        fabric_mask,
        size_px=SWATCH_PX,
        scan_step_px=scan_step_px,
        offset_mode=offset_mode,
    )

    base_name = image_path.stem.replace("_nobg", "")
    save_folder = ensure_dir(output_dir / base_name)
    placements_csv = save_folder / "placements.csv"

    with placements_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "swatch_file",
                "index",
                "left_px",
                "top_px",
                "right_px",
                "bottom_px",
                "size_px",
                "size_cm",
                "preferred_rotation_deg",
            ],
        )
        writer.writeheader()

        for idx, placement in enumerate(placements, start=1):
            crop = image[placement.y : placement.bottom, placement.x : placement.right]
            out_name = f"{base_name}_crop_{idx:02d}.png"
            out_path = save_folder / out_name
            cv2.imwrite(str(out_path), crop)
            writer.writerow(
                {
                    "swatch_file": out_name,
                    "index": idx,
                    "left_px": placement.x,
                    "top_px": placement.y,
                    "right_px": placement.right,
                    "bottom_px": placement.bottom,
                    "size_px": placement.size_px,
                    "size_cm": f"{SWATCH_CM:.2f}",
                    "preferred_rotation_deg": 0,
                }
            )

    save_visualization(image, placements, output_dir / f"{base_name}_packing_vis.png")
    cv2.imwrite(str(output_dir / f"{base_name}_occupied_mask.png"), occupied.astype(np.uint8) * 255)

    fabric_area_px = int(fabric_mask.sum())
    packed_area_px = len(placements) * SWATCH_PX * SWATCH_PX
    coverage_ratio = packed_area_px / fabric_area_px if fabric_area_px else 0.0

    return {
        "src_image": image_path.name,
        "image_width_px": w,
        "image_height_px": h,
        "swatch_size_px": SWATCH_PX,
        "swatch_size_cm": f"{SWATCH_CM:.2f}",
        "scan_step_px": scan_step_px,
        "offset_mode": offset_mode,
        "best_offset_x": best_offset[0],
        "best_offset_y": best_offset[1],
        "num_swatches": len(placements),
        "fabric_area_px": fabric_area_px,
        "packed_area_px": packed_area_px,
        "coverage_ratio": f"{coverage_ratio:.6f}",
        "coverage_area_cm2": f"{(packed_area_px / (SWATCH_PX * SWATCH_PX)) * (SWATCH_CM * SWATCH_CM):.2f}",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    image_files = [path for path in sorted_image_files(input_dir) if path.name.endswith("_nobg.png")]
    print(f"총 {len(image_files)}개의 배경 제거 이미지 발견")
    if not image_files:
        return 0

    summary_path = output_dir / "packing_summary.csv"
    fieldnames = [
        "src_image",
        "image_width_px",
        "image_height_px",
        "swatch_size_px",
        "swatch_size_cm",
        "scan_step_px",
        "offset_mode",
        "best_offset_x",
        "best_offset_y",
        "num_swatches",
        "fabric_area_px",
        "packed_area_px",
        "coverage_ratio",
        "coverage_area_cm2",
        "status",
        "error",
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, image_path in enumerate(image_files, start=1):
            try:
                row = pack_swatches_for_image(
                    image_path,
                    output_dir,
                    scan_step_px=args.scan_step_px,
                    offset_mode=args.offset_mode,
                    alpha_threshold=args.alpha_threshold,
                )
                row.update({"status": "ok", "error": ""})
                writer.writerow(row)
                print(
                    f"[{idx}/{len(image_files)}] 완료: {image_path.name} -> "
                    f"{row['num_swatches']} swatches"
                )
            except Exception as exc:
                writer.writerow(
                    {
                        "src_image": image_path.name,
                        "image_width_px": "",
                        "image_height_px": "",
                        "swatch_size_px": SWATCH_PX,
                        "swatch_size_cm": f"{SWATCH_CM:.2f}",
                        "scan_step_px": args.scan_step_px,
                        "offset_mode": args.offset_mode,
                        "best_offset_x": "",
                        "best_offset_y": "",
                        "num_swatches": 0,
                        "fabric_area_px": "",
                        "packed_area_px": "",
                        "coverage_ratio": "",
                        "coverage_area_cm2": "",
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"[{idx}/{len(image_files)}] 실패: {image_path.name} | {exc}")

    print(f"✅ packing stage finished -> {summary_path}")
    print(f"참고 스케일: 1 swatch = {SWATCH_PX}px ≈ {SWATCH_CM:.2f}cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
