# -*- coding: utf-8 -*-
"""Build mixed weave renders by pairing two swatch folders."""

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_INPUT_DIR = PIPELINE_DIR / "work" / "max_packing"
DEFAULT_BASE_OUTPUT_DIR = PIPELINE_DIR / "outputs" / "mix"
DEFAULT_SWATCH_RES_PX = 376
DEFAULT_NUM_SAMPLES = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_BLENDER_SCRIPT = Path(__file__).resolve().with_name("Mix_Blender.py")


@dataclass(frozen=True)
class PipelineConfig:
    root_input_dir: Path
    base_output_dir: Path
    blender_script: Path
    blender_executable: Path | None = None
    num_samples: int = DEFAULT_NUM_SAMPLES
    swatch_res_px: int = DEFAULT_SWATCH_RES_PX
    random_seed: int | None = DEFAULT_RANDOM_SEED

    def validate(self) -> "PipelineConfig":
        if not self.root_input_dir.exists():
            raise FileNotFoundError(f"Input root does not exist: {self.root_input_dir}")
        if not self.root_input_dir.is_dir():
            raise NotADirectoryError(f"Input root is not a directory: {self.root_input_dir}")
        if not self.blender_script.exists():
            raise FileNotFoundError(f"Blender script does not exist: {self.blender_script}")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be greater than 0")
        if self.swatch_res_px <= 0:
            raise ValueError("swatch_res_px must be greater than 0")
        if self.blender_executable and not self.blender_executable.exists():
            raise FileNotFoundError(
                f"Requested Blender executable does not exist: {self.blender_executable}"
            )

        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        return self


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-input-dir", type=Path, default=DEFAULT_ROOT_INPUT_DIR)
    parser.add_argument("--base-output-dir", type=Path, default=DEFAULT_BASE_OUTPUT_DIR)
    parser.add_argument("--blender-script", type=Path, default=DEFAULT_BLENDER_SCRIPT)
    parser.add_argument("--blender-executable", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--swatch-res-px", type=int, default=DEFAULT_SWATCH_RES_PX)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Disable deterministic sampling by ignoring the seed.",
    )
    args = parser.parse_args()

    return PipelineConfig(
        root_input_dir=args.root_input_dir.resolve(),
        base_output_dir=args.base_output_dir.resolve(),
        blender_script=args.blender_script.resolve(),
        blender_executable=args.blender_executable.resolve()
        if args.blender_executable
        else None,
        num_samples=args.num_samples,
        swatch_res_px=args.swatch_res_px,
        random_seed=None if args.randomize else args.seed,
    )


def calculate_optimal_grid(num_files: int) -> tuple[int, int]:
    if num_files <= 0:
        return 0, 0

    best_cols, best_rows = 1, num_files
    best_score = (abs(best_rows - best_cols), best_cols * best_rows - num_files, best_cols * best_rows)

    for cols in range(1, int(math.sqrt(num_files)) + 2):
        rows = math.ceil(num_files / cols)
        score = (abs(rows - cols), cols * rows - num_files, cols * rows)
        if score < best_score:
            best_cols, best_rows, best_score = cols, rows, score

    return best_cols, best_rows


def collect_subfolders(root_dir: Path) -> list[Path]:
    return sorted(path for path in root_dir.iterdir() if path.is_dir())


def collect_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def filter_readable_images(image_paths: Sequence[Path]) -> tuple[list[Path], list[str]]:
    readable: list[Path] = []
    failures: list[str] = []

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                image.verify()
            readable.append(image_path)
        except (OSError, UnidentifiedImageError) as exc:
            failures.append(f"{image_path.name}: {exc}")

    return readable, failures


def create_texture_from_folder(
    input_dir: Path,
    save_path: Path,
    swatch_res_px: int,
    rng: random.Random,
    req_grid: tuple[int, int] | None = None,
) -> tuple[int, int]:
    image_paths = collect_image_paths(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {input_dir}")

    readable_images, failures = filter_readable_images(image_paths)
    if failures:
        print(f"   Skipped {len(failures)} unreadable images in {input_dir.name}")
    if not readable_images:
        raise RuntimeError(f"All images in {input_dir} were unreadable")

    if req_grid is None:
        cols, rows = calculate_optimal_grid(len(readable_images))
    else:
        cols, rows = req_grid

    if cols <= 0 or rows <= 0:
        raise ValueError(f"Invalid grid size requested: {cols}x{rows}")

    target_count = cols * rows
    chosen_images = list(readable_images)
    rng.shuffle(chosen_images)
    tiled_images = [chosen_images[index % len(chosen_images)] for index in range(target_count)]

    canvas = Image.new("RGBA", (cols * swatch_res_px, rows * swatch_res_px))
    for index, image_path in enumerate(tiled_images):
        row, col = divmod(index, cols)
        with Image.open(image_path) as image:
            patch = image.convert("RGBA")
            if patch.size != (swatch_res_px, swatch_res_px):
                patch = patch.resize((swatch_res_px, swatch_res_px), Image.LANCZOS)
            canvas.paste(patch, (col * swatch_res_px, row * swatch_res_px))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)
    return cols, rows


def resolve_blender_executable(requested_path: Path | None = None) -> Path:
    if requested_path is not None:
        return requested_path

    env_blender = os.environ.get("BLENDER_EXE")
    if env_blender:
        env_path = Path(env_blender)
        if env_path.exists():
            return env_path

    shell_blender = shutil.which("blender")
    if shell_blender:
        return Path(shell_blender)

    install_root = Path(r"C:\Program Files\Blender Foundation")
    if install_root.exists():
        for candidate_dir in sorted(install_root.iterdir(), reverse=True):
            blender_exe = candidate_dir / "blender.exe"
            if blender_exe.exists():
                return blender_exe

    raise FileNotFoundError("Blender executable could not be found automatically")


def run_blender_mix(
    config: PipelineConfig,
    warp_texture: Path,
    weft_texture: Path,
    cols: int,
    rows: int,
    output_path: Path,
) -> None:
    blender_executable = resolve_blender_executable(config.blender_executable)
    command = [
        str(blender_executable),
        "-b",
        "-P",
        str(config.blender_script),
        "--",
        str(warp_texture),
        str(weft_texture),
        str(cols),
        str(rows),
        str(output_path),
    ]

    completed = subprocess.run(command, cwd=str(config.blender_script.parent), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Blender render failed with exit code {completed.returncode} for {output_path.name}"
        )


def build_folder_pairs(
    folders: Sequence[Path],
    num_samples: int,
    rng: random.Random,
) -> list[tuple[Path, Path]]:
    pairs = list(itertools.permutations(folders, 2))
    if not pairs:
        return []

    rng.shuffle(pairs)
    if num_samples <= len(pairs):
        return pairs[:num_samples]

    selected_pairs = list(pairs)
    while len(selected_pairs) < num_samples:
        selected_pairs.append(rng.choice(pairs))
    return selected_pairs


def run_pipeline(config: PipelineConfig) -> int:
    rng = random.Random(config.random_seed)
    folders = collect_subfolders(config.root_input_dir)
    if len(folders) < 2:
        raise RuntimeError("At least two source folders are required for mix generation")

    selected_pairs = build_folder_pairs(folders, config.num_samples, rng)
    print(f"========== Mix generation start ({len(selected_pairs)} samples) ==========")
    print(f"Source root: {config.root_input_dir}")
    print(f"Output root: {config.base_output_dir}")
    if config.random_seed is not None:
        print(f"Seed: {config.random_seed}")
    else:
        print("Seed: random")

    success_count = 0
    failure_count = 0

    for sample_index, (warp_folder, weft_folder) in enumerate(selected_pairs, start=1):
        mix_name = f"Mix_{sample_index:02d}_W({warp_folder.name})_x_H({weft_folder.name})"
        output_dir = config.base_output_dir / mix_name
        warp_texture_path = output_dir / "texture_warp.png"
        weft_texture_path = output_dir / "texture_weft.png"
        render_output_path = output_dir / f"{mix_name}.png"

        print(f"\n[{sample_index}/{len(selected_pairs)}] {mix_name}")

        try:
            cols, rows = create_texture_from_folder(
                warp_folder,
                warp_texture_path,
                config.swatch_res_px,
                rng,
            )
            create_texture_from_folder(
                weft_folder,
                weft_texture_path,
                config.swatch_res_px,
                rng,
                req_grid=(cols, rows),
            )
            print(f"   Render grid: {cols}x{rows}")
            run_blender_mix(
                config,
                warp_texture_path,
                weft_texture_path,
                cols,
                rows,
                render_output_path,
            )
            success_count += 1
            print("   Completed")
        except Exception as exc:
            failure_count += 1
            print(f"   Failed: {exc}")

    print("\n========== Mix generation summary ==========")
    print(f"Success: {success_count}")
    print(f"Failed: {failure_count}")
    return 0 if failure_count == 0 else 1


def main() -> int:
    try:
        config = parse_args().validate()
        return run_pipeline(config)
    except Exception as exc:
        print(f"[Error] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
