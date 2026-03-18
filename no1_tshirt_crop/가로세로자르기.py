# -*- coding: utf-8 -*-
# 376x376 이미지를 가로 10등분 + 세로 10등분 ⇒ 총 20장 PNG로 저장
# 필요 라이브러리: Pillow (pip install pillow)

import os
from pathlib import Path
from PIL import Image, ImageOps

# ===== 입력 이미지 경로 =====
INPUT_PATH = r"/no1_tshirt_crop/1022_crop_size_1010/print_1_rotate_crop_rot2.png"

# ===== 출력 폴더 설정 =====
in_path = Path(INPUT_PATH)
out_dir = in_path.parent / "2slices_10h_10v"
out_dir.mkdir(parents=True, exist_ok=True)

# 파일명 안전화
stem = in_path.stem
safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)

def split_sizes(total: int, parts: int):
    """
    total 픽셀을 parts로 정수 분할.
    예: total=376, parts=10 ⇒ [38,38,38,38,38,38,37,37,37,37]
    (앞에서부터 +1 픽셀씩 배분)
    """
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]

def cumulative_starts(sizes):
    """각 조각의 시작 오프셋 리스트 반환 (0, s0, s0+s1, ...)"""
    starts = [0]
    for s in sizes[:-1]:
        starts.append(starts[-1] + s)
    return starts

def main():
    # 이미지 로드 + EXIF 회전 보정
    img = Image.open(in_path)
    img = ImageOps.exif_transpose(img)  # (있으면) EXIF 방향 보정
    w, h = img.size  # 기대: 376 x 376

    # ===== 가로 10등분(수평 밴드 10장: 위→아래) =====
    h_sizes = split_sizes(h, 10)              # 각 밴드 높이
    h_starts = cumulative_starts(h_sizes)     # 각 밴드 시작 y
    for i, (ys, hh) in enumerate(zip(h_starts, h_sizes)):
        # crop box: (left, upper, right, lower)  — 좌상단 원점
        box = (0, ys, w, ys + hh)
        band = img.crop(box)
        out_path = out_dir / f"{safe_stem}_H_{i:02d}.png"
        band.save(out_path, format="PNG")
        print(f"[H] saved: {out_path.name}  size={band.size}")

    # ===== 세로 10등분(수직 밴드 10장: 왼→오) =====
    w_sizes = split_sizes(w, 10)              # 각 밴드 폭
    w_starts = cumulative_starts(w_sizes)     # 각 밴드 시작 x
    for j, (xs, ww) in enumerate(zip(w_starts, w_sizes)):
        box = (xs, 0, xs + ww, h)
        band = img.crop(box)
        out_path = out_dir / f"{safe_stem}_V_{j:02d}.png"
        band.save(out_path, format="PNG")
        print(f"[V] saved: {out_path.name}  size={band.size}")

    print(f"\n[DONE] Output folder: {out_dir}")
    print(f"       Horizontal: 10 files, Vertical: 10 files (total 20)")

if __name__ == "__main__":
    main()
