# -*- coding: utf-8 -*-
"""
배경 제거 → 옷 내부에서 10cm x 10cm 정사각형 스와치 10개 추출 → 각 스와치를 10h/10v로 분할 저장
- 앞 5개: 정방향 저장
- 뒤 5개: 90도 회전 저장
- cm↔px 스케일: 3755px ↔ 100cm (가로 기준, 1cm ≈ 37.55px → 10cm ≈ 376px)
- 마스크 내부에 완전히 들어가도록 Distance Transform 기반 배치 + 서로 겹치지 않게 배치
- 스와치 시각화 이미지(정방향 박스=라임, 회전샘플 박스=노랑) 및 CSV 기록
- 각 스와치 PNG는 slices_root_dir/<스와치파일명>/ 에 H_00~09, V_00~09로 저장(폴더 분리 보관)
"""

import os
import math
import random
from glob import glob
import csv
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps

import onnxruntime as ort
from rembg import remove, new_session

# =========================
# 사용자 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent
input_folder = str(BASE_DIR / "data" / "input_images")
output_folder = str(BASE_DIR / "work" / "cropped")
SLICES_ROOT_DIR = os.path.join(output_folder, "slices_10h_10v")  # ← 분할이미지 보관 폴더(스와치별 하위폴더 생성)

# 샘플 개수
N_NORMAL  = 5   # 정방향
N_ROTATED = 5   # 90도 회전 저장
TOTAL_SAMPLES = N_NORMAL + N_ROTATED

MAX_TRIES_PER_CROP = 120
NON_OVERLAP_GAP_PX = 10
RNG_SEED = 42

# rembg 모델
REMBG_MODEL_NAME = "u2net"

# ======= cm↔px 스케일 (가로 기준 고정) =======
# 3755 px ↔ 100 cm → 1cm = 37.55 px → 10cm ≈ 375.5 px → 376 px
REF_W_PX, REF_W_CM = 3755, 100.0
PX_PER_CM = REF_W_PX / REF_W_CM  # 37.55
SWATCH_CM = 10.0
SWATCH_PX = int(round(SWATCH_CM * PX_PER_CM))  # 10cm 스와치 한 변(px) → 376
print(f"스와치 한 변: {SWATCH_PX}px (≈ {SWATCH_CM}cm)")

# (선택) 파일명에 cm 표기 포함 여부
APPEND_CM_TO_FILENAME = False


# =========================
# 유틸 함수
# =========================
def setup_session():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("▶ rembg: CUDAExecutionProvider 활성화 (GPU 사용)")
    else:
        providers = ["CPUExecutionProvider"]
        print("▶ rembg: CPUExecutionProvider만 사용 (GPU 미사용)")
    return new_session(REMBG_MODEL_NAME, providers=providers)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, ext)))
    files.sort()
    return files


def rect_intersects(r1, r2, gap=0):
    """두 사각형이 gap 포함하여 겹치면 True"""
    l1, t1, r1_, b1 = r1
    l2, t2, r2_, b2 = r2
    return not (r1_ + gap <= l2 or r2_ + gap <= l1 or b1 + gap <= t2 or b2 + gap <= t1)


# ======= (분할) 보조 함수 =======
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


def split_and_save_10h10v(img_path: Path, slices_root_dir: Path):
    """
    스와치 PNG(정방향/회전)를 받아서 가로10, 세로10 분할 PNG를 저장.
    저장 폴더: slices_root_dir / <스와치파일명>
    파일명: <스와치파일명>_H_##.png, _V_##.png
    """
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # 혹시 모를 EXIF 방향 보정
    w, h = img.size

    # 파일명 안전화 및 개별 폴더 생성
    stem = img_path.stem
    safe_stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)
    out_dir = Path(slices_root_dir) / safe_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # 수평 밴드 10장 (위→아래)
    h_sizes = split_sizes(h, 10)
    h_starts = cumulative_starts(h_sizes)
    for i, (ys, hh) in enumerate(zip(h_starts, h_sizes)):
        box = (0, ys, w, ys + hh)
        band = img.crop(box)
        out_path = out_dir / f"{safe_stem}_H_{i:02d}.png"
        band.save(out_path, format="PNG")
        print(f"[H] saved: {out_path}")

    # 수직 밴드 10장 (왼→오)
    w_sizes = split_sizes(w, 10)
    w_starts = cumulative_starts(w_sizes)
    for j, (xs, ww) in enumerate(zip(w_starts, w_sizes)):
        box = (xs, 0, xs + ww, h)
        band = img.crop(box)
        out_path = out_dir / f"{safe_stem}_V_{j:02d}.png"
        band.save(out_path, format="PNG")
        print(f"[V] saved: {out_path}")

    print(f"    ↳ 분할 저장 위치: {out_dir}")


# =========================
# 메인
# =========================
def main():
    if RNG_SEED is not None:
        random.seed(RNG_SEED)
        np.random.seed(RNG_SEED)

    ensure_dir(output_folder)
    ensure_dir(SLICES_ROOT_DIR)  # 분할이미지 루트 폴더

    # CSV 준비
    csv_path = os.path.join(output_folder, "crop_sizes.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_f)
    csv_writer.writerow([
        "src_image", "crop_index", "rotated_90deg",
        "left_px", "top_px", "right_px", "bottom_px",
        "width_px", "height_px",
        "width_cm", "height_cm", "area_cm2"
    ])

    files = list_images(input_folder)
    print(f"총 {len(files)}개 이미지 처리 시작...")

    session = setup_session()

    for idx, file in enumerate(files, 1):
        try:
            # 1) 이미지 열기 (RGBA)
            input_image = Image.open(file).convert("RGBA")
            # 2) 배경 제거
            output_image = remove(input_image, session=session)

            file_name = os.path.basename(file)
            base_name = os.path.splitext(file_name)[0]

            # 3) 알파 마스크 & bbox 크롭
            mask = output_image.getchannel("A")
            bbox = mask.getbbox()
            if bbox is None:
                print(f"[{idx}] bbox 없음 (옷 영역 인식 실패): {file}")
                continue

            output_image = output_image.crop(bbox)
            mask = mask.crop(bbox)

            # 배경제거 결과 저장
            nobg_path = os.path.join(output_folder, f"{base_name}_nobg.png")
            output_image.save(nobg_path)
            print(f"[{idx}/{len(files)}] 완료: {nobg_path}")

            # 4) 마스크 이진화
            mask_array = np.array(mask)
            mask_bin = (mask_array > 0).astype(np.uint8) * 255

            # 5) 시각화용 마스크 윤곽
            mask_vis = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_vis, contours, -1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), mask_vis)

            # 6) Distance Transform
            dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3).astype(np.float32)

            # 7) 스와치 배치
            w, h = output_image.size
            placed_rects = []
            vis_image = output_image.copy()
            draw = ImageDraw.Draw(vis_image)

            crop_w = crop_h = SWATCH_PX
            radius = 0.5 * math.hypot(crop_w, crop_h)  # 정사각형 대각선의 절반

            # 공통 후보 좌표 (사각형이 완전히 들어갈 수 있는 중심점)
            ys, xs = np.where(dist >= radius)
            valid_centers = list(zip(xs, ys))  # (cx, cy)

            if not valid_centers:
                print(f"   └ [경고] 유효 중심 후보 없음: {file_name}")
                continue

            # 5개 정방향 + 5개 회전 샘플
            sample_plan = (
                [("normal", i + 1) for i in range(N_NORMAL)] +
                [("rot90",  i + 1) for i in range(N_ROTATED)]
            )

            for kind, local_idx in sample_plan:
                success = False
                for attempt in range(MAX_TRIES_PER_CROP):
                    cx, cy = random.choice(valid_centers)
                    left   = cx - crop_w // 2
                    top    = cy - crop_h // 2
                    right  = left + crop_w
                    bottom = top + crop_h

                    # 경계 체크
                    if left < 0 or top < 0 or right > w or bottom > h:
                        continue

                    rect = (left, top, right, bottom)

                    # 기존 박스와 비겹침
                    if any(rect_intersects(rect, r, NON_OVERLAP_GAP_PX) for r in placed_rects):
                        continue

                    # 통과 → 크롭
                    cropped = output_image.crop(rect)

                    # 회전 여부
                    rotated_flag = (kind == "rot90")
                    if rotated_flag:
                        cropped_to_save = cropped.rotate(90, expand=False)
                    else:
                        cropped_to_save = cropped

                    # 파일명
                    idx_tag = local_idx  # 1~5
                    if rotated_flag:
                        fname = f"{base_name}_crop_rot{idx_tag}.png"
                    else:
                        fname = f"{base_name}_crop{idx_tag}.png"

                    if APPEND_CM_TO_FILENAME:
                        fname = fname.replace(".png", f"_{SWATCH_CM:.0f}cm.png")

                    save_path = os.path.join(output_folder, fname)
                    cropped_to_save.save(save_path)

                    # 시각화 박스(정방향=라임, 회전=노랑)
                    outline_color = "lime" if not rotated_flag else "yellow"
                    draw.rectangle(rect, outline=outline_color, width=3)
                    draw.text((left + 4, top + 4),
                              f"{SWATCH_CM:.0f}×{SWATCH_CM:.0f}cm",
                              fill=outline_color)

                    # CSV 기록
                    width_px = crop_w
                    height_px = crop_h
                    width_cm = SWATCH_CM
                    height_cm = SWATCH_CM
                    area_cm2 = width_cm * height_cm

                    csv_writer.writerow([
                        file_name,
                        f"{'rot' if rotated_flag else 'norm'}-{idx_tag}",
                        1 if rotated_flag else 0,
                        left, top, right, bottom,
                        width_px, height_px,
                        f"{width_cm:.2f}", f"{height_cm:.2f}",
                        f"{area_cm2:.2f}"
                    ])

                    print(f"   └ 저장: {save_path} | "
                          f"{width_px}x{height_px}px ≈ {width_cm:.1f}x{height_cm:.1f}cm (회전:{rotated_flag})")
                    placed_rects.append(rect)

                    # ★ 추가 단계: 방금 저장한 스와치를 10h/10v로 분할 저장 (스와치별 개별 폴더)
                    split_and_save_10h10v(Path(save_path), Path(SLICES_ROOT_DIR))

                    success = True
                    break

                if not success:
                    print(f"   └ [경고] {kind} #{local_idx} 배치 실패 (적합 위치 부족)")

            # 스와치 배치 시각화 이미지 저장
            vis_path = os.path.join(output_folder, f"{base_name}_visual.png")
            vis_image.save(vis_path)

        except Exception as e:
            print(f"[{idx}/{len(files)}] 실패: {file}, 이유: {e}")

    csv_f.close()
    print("✅ 완료: (1) 배경제거, (2) 10cm 스와치 10개 저장, (3) 각 스와치 10h/10v 분할 저장(폴더별 보관), (4) CSV/시각화 저장")


if __name__ == "__main__":
    main()
