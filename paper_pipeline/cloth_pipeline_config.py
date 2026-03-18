from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

# -----------------------------------------------------------------------------
# Shared physical scale
# -----------------------------------------------------------------------------
REF_W_PX: int = 3755
REF_W_CM: float = 100.0
PX_PER_CM: float = REF_W_PX / REF_W_CM
SWATCH_CM: float = 10.0
SWATCH_PX: int = int(round(SWATCH_CM * PX_PER_CM))

# -----------------------------------------------------------------------------
# Shared processing parameters
# -----------------------------------------------------------------------------
ALPHA_THRESHOLD: int = 10
RANDOM_SEED: int = 42
VALID_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

# Greedy packing
PACK_SCAN_STEP_PX: int = 10
PACK_OFFSET_MODE: str = "coarse"  # single | coarse | full

# Segmentation
REMBG_MODEL_NAME: str = "u2net"

# Optional slicing
SLICE_PARTS: int = 10


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sorted_image_files(folder: Path | str, *, recursive: bool = False) -> List[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []

    if recursive:
        candidates = folder.rglob("*")
    else:
        candidates = folder.iterdir()

    files = [
        path
        for path in candidates
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.as_posix().lower())


def split_sizes(total: int, parts: int) -> List[int]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    base = total // parts
    remainder = total % parts
    return [base + (1 if idx < remainder else 0) for idx in range(parts)]


def cumulative_starts(sizes: Iterable[int]) -> List[int]:
    starts: List[int] = []
    offset = 0
    for size in sizes:
        starts.append(offset)
        offset += int(size)
    return starts


def cm_to_px(cm: float) -> int:
    return int(round(cm * PX_PER_CM))


def px_to_cm(px: int | float) -> float:
    return float(px) / PX_PER_CM
