# -*- coding: utf-8 -*-
"""
[LEGACY BASELINE] Raster-scan greedy packing.

This is the original swatch extraction code. It scans the image from the top-left
in row-major order with a 10 px stride and immediately accepts every 376 x 376 px
window that lies fully inside the garment mask and does not overlap an already
accepted window. Its placement is decided by the scan order, so it does NOT
guarantee a maximum swatch count; in the paper it is only the "raster greedy"
baseline of the placement comparison (Section 4.4, Table 2, Figures 9-10).

The final method of the paper is 2_extract_swatches.py (grid-origin optimization).

Inputs : work/01_segmented/*_nobg.png
Outputs: work/legacy_greedy_packing/<garment>/<garment>_crop_NN.png
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_config import SEGMENT_DIR, ROOT  # noqa: E402

# ====== 사용자 설정 ======
INPUT_DIR = str(SEGMENT_DIR)                                   # *_nobg.png from step 1
OUTPUT_DIR = str(ROOT / "work" / "legacy_greedy_packing")     # baseline output (not used downstream)

# 물리적 기준 (고정)
REF_W_PX = 3755  # 100cm
REF_W_CM = 100.0
PX_PER_CM = REF_W_PX / REF_W_CM  # 37.55 px
SWATCH_CM = 10.0
SWATCH_PX = int(round(SWATCH_CM * PX_PER_CM))  # 약 376 px

# 검색 정밀도 (작을수록 더 꼼꼼하게 찾지만 느려짐)
# 10px 단위로 이동하며 자리를 찾습니다. (충분히 꼼꼼함)
SCAN_STEP_PX = 10


def pack_swatches_tightly(img_path):
    img_path = Path(img_path)
    # _nobg.png 파일만 처리 (알파 채널 필요)
    if "nobg" not in img_path.name:
        return

    # 1. 이미지 읽기 (OpenCV 사용)
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None: return

    # 알파 채널 확인
    if img.shape[2] < 4:
        print(f"⚠️ 알파 채널 없음: {img_path.name}")
        return

    h, w = img.shape[:2]
    alpha = img[:, :, 3]

    # 2. '사용됨' 마스크 생성 (0: 빈공간, 1: 이미 스와치 뽑음)
    occupied_mask = np.zeros((h, w), dtype=np.uint8)

    # 원본 알파 마스크 이진화 (옷이 있는 곳=255, 없는 곳=0)
    # 노이즈 제거를 위해 약간의 threshold 적용
    _, fabric_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    swatch_count = 0
    save_folder = Path(OUTPUT_DIR) / img_path.stem.replace("_nobg", "")
    save_folder.mkdir(parents=True, exist_ok=True)

    print(f"[{img_path.stem}] 꼼꼼한 채우기 시작... (Box: {SWATCH_PX}px)")

    # 3. 전체 스캔 (왼쪽->오른쪽, 위->아래)
    # 테트리스 하듯이 자리를 찾습니다.
    for y in range(0, h - SWATCH_PX, SCAN_STEP_PX):
        for x in range(0, w - SWATCH_PX, SCAN_STEP_PX):

            # (A) 이 자리가 이미 사용되었는지 확인 (중심점 or 네 귀퉁이 체크보다 확실하게 영역 체크)
            # 속도를 위해 슬라이싱으로 체크: occupied_mask 영역에 1이 하나라도 있으면 스킵
            if np.any(occupied_mask[y:y + SWATCH_PX, x:x + SWATCH_PX]):
                continue

            # (B) 이 자리가 옷 내부인지 확인
            # 해당 영역의 fabric_mask가 모두 255여야 함 (구멍 없어야 함)
            # 즉, 최소값이 0이면 안됨 (0은 빈공간)
            patch_mask = fabric_mask[y:y + SWATCH_PX, x:x + SWATCH_PX]
            if cv2.countNonZero(patch_mask) < (SWATCH_PX * SWATCH_PX):
                # 꽉 차있지 않음 (가장자리거나 구멍)
                continue

            # === 찾았다! (유효하고 & 빈 자리) ===
            swatch_count += 1

            # 1. 자리 선점 (마킹)
            occupied_mask[y:y + SWATCH_PX, x:x + SWATCH_PX] = 1

            # 2. 이미지 크롭 및 저장
            crop = img[y:y + SWATCH_PX, x:x + SWATCH_PX]

            # 파일명: 원본명_crop_01.png
            out_name = f"{img_path.stem.replace('_nobg', '')}_crop_{swatch_count:02d}.png"
            cv2.imwrite(str(save_folder / out_name), crop)

            # (선택) 시각화: 원본 이미지에 빨간 박스 그리기 (디버깅용)
            # cv2.rectangle(img, (x, y), (x+SWATCH_PX, y+SWATCH_PX), (0, 0, 255), 2)

    print(f"   ✅ 총 {swatch_count}개의 스와치 생성 완료! -> {save_folder}")

    # 결과 시각화 이미지 저장 (어디서 뽑았는지 확인용)
    # 사용된 영역을 시각적으로 보여줌
    vis_path = Path(OUTPUT_DIR) / f"{img_path.stem}_packing_vis.jpg"
    # occupied_mask를 보기 좋게 변환 (0/1 -> 0/255)
    vis_img = occupied_mask * 255
    cv2.imwrite(str(vis_path), vis_img)


def main():
    if not os.path.exists(INPUT_DIR):
        print("입력 폴더가 없습니다.")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith("_nobg.png")]
    print(f"총 {len(files)}개의 배경 제거 이미지 발견.")

    for f in files:
        pack_swatches_tightly(os.path.join(INPUT_DIR, f))


if __name__ == "__main__":
    main()
