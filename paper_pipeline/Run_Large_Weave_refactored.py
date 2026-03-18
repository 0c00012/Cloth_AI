# -*- coding: utf-8 -*-
"""Refactored 2D texture assembly + Blender handoff.

Key changes:
- Uses every packed swatch exactly once.
- Keeps deterministic ordering and seed-controlled shuffling.
- Reads optional rotation metadata from placements.csv.
- Checks Blender exit status and saves render logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image

from cloth_pipeline_config import RANDOM_SEED, SWATCH_PX, ensure_dir, sorted_image_files


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble master textures and launch Blender rendering.")
    parser.add_argument("--root-input-dir", required=True, help="Root folder containing packed swatch subfolders.")
    parser.add_argument("--base-output-dir", required=True, help="Root folder for texture and render outputs.")
    parser.add_argument(
        "--blender-script",
        default="Blender_Large_Weave_refactored.py",
        help="Blender Python script filename or absolute path.",
    )
    parser.add_argument(
        "--blender-exe",
        default="",
        help="Optional explicit path to Blender executable. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for tile shuffling. Default: {RANDOM_SEED}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing texture/render files.",
    )
    return parser.parse_args(argv)


def choose_near_square_grid(num_tiles: int) -> tuple[int, int]:
    if num_tiles <= 0:
        return 0, 0
    cols = math.ceil(math.sqrt(num_tiles))
    rows = math.ceil(num_tiles / cols)
    return cols, rows


def load_rotation_metadata(folder: Path) -> Dict[str, int]:
    csv_path = folder / "placements.csv"
    if not csv_path.exists():
        return {}

    mapping: Dict[str, int] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                name = row.get("swatch_file", "").strip()
                rotation = int(float(row.get("preferred_rotation_deg", 0)))
            except ValueError:
                continue
            if name:
                mapping[name] = rotation % 360
    return mapping


def gather_swatches(folder: Path) -> List[Path]:
    return [path for path in sorted_image_files(folder) if path.name != "placements.csv"]


def create_master_texture(folder: Path, save_path: Path, random_seed: int) -> dict:
    swatches = gather_swatches(folder)
    num_files = len(swatches)
    if num_files == 0:
        raise ValueError(f"No swatch images found in {folder}")

    cols, rows = choose_near_square_grid(num_files)
    texture_width = cols * SWATCH_PX
    texture_height = rows * SWATCH_PX
    canvas = Image.new("RGBA", (texture_width, texture_height), (0, 0, 0, 0))

    rotation_map = load_rotation_metadata(folder)
    rng = random.Random(random_seed)
    ordered_swatches = list(swatches)
    rng.shuffle(ordered_swatches)

    tile_records = []
    for tile_index, swatch_path in enumerate(ordered_swatches):
        col = tile_index % cols
        row = tile_index // cols
        x = col * SWATCH_PX
        y = row * SWATCH_PX

        patch = Image.open(swatch_path).convert("RGBA")
        if patch.size != (SWATCH_PX, SWATCH_PX):
            patch = patch.resize((SWATCH_PX, SWATCH_PX), Image.LANCZOS)

        rotation_deg = rotation_map.get(swatch_path.name, 0)
        if rotation_deg:
            patch = patch.rotate(rotation_deg, expand=False)

        canvas.paste(patch, (x, y), patch)
        tile_records.append(
            {
                "tile_index": tile_index,
                "row": row,
                "col": col,
                "x_px": x,
                "y_px": y,
                "source_file": swatch_path.name,
                "rotation_deg": rotation_deg,
            }
        )

    ensure_dir(save_path.parent)
    canvas.save(save_path)

    empty_slots = cols * rows - num_files
    layout_metadata = {
        "source_folder": folder.name,
        "random_seed": random_seed,
        "num_swatches": num_files,
        "grid_cols": cols,
        "grid_rows": rows,
        "empty_slots": empty_slots,
        "swatch_px": SWATCH_PX,
        "texture_width_px": texture_width,
        "texture_height_px": texture_height,
        "tiles": tile_records,
    }
    metadata_path = save_path.with_name(f"{save_path.stem}_layout.json")
    metadata_path.write_text(json.dumps(layout_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return layout_metadata


def find_blender_auto(explicit_path: str = "") -> Optional[str]:
    if explicit_path:
        path = Path(explicit_path)
        return str(path) if path.exists() else None

    path = shutil.which("blender")
    if path:
        return path

    candidate_roots = [
        Path(r"C:\Program Files\Blender Foundation"),
        Path("/Applications/Blender.app/Contents/MacOS"),
    ]
    for root in candidate_roots:
        if not root.exists():
            continue
        if root.is_file():
            return str(root)
        children = sorted(root.iterdir(), reverse=True)
        for child in children:
            if child.is_file() and child.name.lower().startswith("blender"):
                return str(child)
            exe = child / "blender.exe"
            if exe.exists():
                return str(exe)
            mac_exe = child / "Blender"
            if mac_exe.exists():
                return str(mac_exe)
    return None


def run_blender(
    *,
    blender_exe: str,
    blender_script: Path,
    texture_path: Path,
    cols: int,
    rows: int,
    texture_width_px: int,
    texture_height_px: int,
    render_out_path: Path,
    log_path: Path,
) -> None:
    cmd = [
        blender_exe,
        "-b",
        "-P",
        str(blender_script),
        "--",
        str(texture_path),
        str(cols),
        str(rows),
        str(render_out_path),
        str(texture_width_px),
        str(texture_height_px),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log_text = "[STDOUT]\n" + result.stdout + "\n[STDERR]\n" + result.stderr
    log_path.write_text(log_text, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Blender render failed with return code {result.returncode}. See log: {log_path}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root_input_dir = Path(args.root_input_dir)
    base_output_dir = ensure_dir(args.base_output_dir)

    if not root_input_dir.exists():
        raise SystemExit(f"Root input directory does not exist: {root_input_dir}")

    this_dir = Path(__file__).resolve().parent
    blender_script = Path(args.blender_script)
    if not blender_script.is_absolute():
        blender_script = this_dir / blender_script
    if not blender_script.exists():
        raise SystemExit(f"Blender script not found: {blender_script}")

    blender_exe = find_blender_auto(args.blender_exe)
    if not blender_exe:
        raise SystemExit("Unable to locate Blender executable.")

    swatch_folders = sorted([path for path in root_input_dir.iterdir() if path.is_dir()], key=lambda p: p.name.lower())
    print(f"발견된 스와치 폴더 수: {len(swatch_folders)}")

    summary_path = base_output_dir / "render_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "folder",
                "num_swatches",
                "grid_cols",
                "grid_rows",
                "empty_slots",
                "texture_file",
                "render_file",
                "status",
                "error",
            ],
        )
        writer.writeheader()

        for idx, folder in enumerate(swatch_folders, start=1):
            output_folder = ensure_dir(base_output_dir / folder.name)
            texture_path = output_folder / f"{folder.name}_texture.png"
            render_path = output_folder / f"{folder.name}_render.png"
            log_path = output_folder / "blender_render.log"

            if not args.overwrite and texture_path.exists() and render_path.exists():
                writer.writerow(
                    {
                        "folder": folder.name,
                        "num_swatches": "",
                        "grid_cols": "",
                        "grid_rows": "",
                        "empty_slots": "",
                        "texture_file": texture_path.name,
                        "render_file": render_path.name,
                        "status": "skipped_existing",
                        "error": "",
                    }
                )
                print(f"[{idx}/{len(swatch_folders)}] 건너뜀: {folder.name} (existing outputs)")
                continue

            try:
                layout = create_master_texture(folder, texture_path, args.random_seed)
                run_blender(
                    blender_exe=blender_exe,
                    blender_script=blender_script,
                    texture_path=texture_path,
                    cols=layout["grid_cols"],
                    rows=layout["grid_rows"],
                    texture_width_px=layout["texture_width_px"],
                    texture_height_px=layout["texture_height_px"],
                    render_out_path=render_path,
                    log_path=log_path,
                )
                writer.writerow(
                    {
                        "folder": folder.name,
                        "num_swatches": layout["num_swatches"],
                        "grid_cols": layout["grid_cols"],
                        "grid_rows": layout["grid_rows"],
                        "empty_slots": layout["empty_slots"],
                        "texture_file": texture_path.name,
                        "render_file": render_path.name,
                        "status": "ok",
                        "error": "",
                    }
                )
                print(
                    f"[{idx}/{len(swatch_folders)}] 완료: {folder.name} -> "
                    f"grid {layout['grid_cols']}x{layout['grid_rows']}"
                )
            except Exception as exc:
                writer.writerow(
                    {
                        "folder": folder.name,
                        "num_swatches": 0,
                        "grid_cols": 0,
                        "grid_rows": 0,
                        "empty_slots": 0,
                        "texture_file": texture_path.name,
                        "render_file": render_path.name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                print(f"[{idx}/{len(swatch_folders)}] 실패: {folder.name} | {exc}")

    print(f"✅ render pipeline finished -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
