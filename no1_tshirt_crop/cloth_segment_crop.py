# -*- coding: utf-8 -*-
import os
import math
import random
from glob import glob

import numpy as np
import cv2
from PIL import Image, ImageDraw

# rembg + onnxruntime (GPU 가용 시 CUDA 사용)
import onnxruntime as ort
from rembg import remove, new_session


# =========================
# 사용자 설정
# =========================
# 입력/출력 경로
input_folder = r"../data/0819_data"
output_folder = r"../no1_tshirt_crop/0819_JH_crop"

# 크롭 설정
CROP_COUNT_PER_IMAGE = 5                 # 이미지당 크롭 개수
CROP_RATIO_MIN, CROP_RATIO_MAX = 0.10, 0.40  # 크롭 박스 너비/높이 비율 범위
MAX_TRIES_PER_CROP = 120                 # 각 크롭 박스 배치 시도 횟수
NON_OVERLAP_GAP_PX = 10                  # 크롭 박스 사이 최소 간격(px)
RNG_SEED = 42                            # 재현성용 시드(원치 않으면 None)

# rembg 모델 선택: 'u2net', 'u2netp', 'u2net_human_seg', 'isnet-general-use' 등
REMBG_MODEL_NAME = "u2net"


# =========================
# 유틸 함수
# =========================
def setup_session():
    """
    onnxruntime-gpu 가 설치되어 있고 CUDA 사용 가능하면 GPU 우선으로,
    아니면 CPU 로 rembg 세션 생성
    """
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("▶ rembg: CUDAExecutionProvider 활성화 (GPU 사용)")
    else:
        providers = ["CPUExecutionProvider"]
        print("▶ rembg: CPUExecutionProvider만 사용 (GPU 미사용)")

    # rembg 세션 생성 (providers 우선순위 지정)
    session = new_session(REMBG_MODEL_NAME, providers=providers)
    return session


def rect_intersects(r1, r2, gap=0):
    """ 두 사각형이 겹치는지 여부 체크 (gap 적용) """
    l1, t1, r1_, b1 = r1
    l2, t2, r2_, b2 = r2
    return not (r1_ + gap <= l2 or r2_ + gap <= l1 or b1 + gap <= t2 or b2 + gap <= t1)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_images(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, ext)))
    files.sort()
    return files


# =========================
# 메인 로직
# =========================
def main():
    if RNG_SEED is not None:
        random.seed(RNG_SEED)
        np.random.seed(RNG_SEED)

    ensure_dir(output_folder)

    files = list_images(input_folder)
    print(f"총 {len(files)}개 이미지 처리 시작...")

    # rembg 세션 (GPU 가능 시 GPU)
    session = setup_session()

    for idx, file in enumerate(files, 1):
        try:
            # 1) 이미지 열기 (RGBA)
            input_image = Image.open(file).convert("RGBA")

            # 2) 배경 제거 (GPU/CPU)
            output_image = remove(input_image, session=session)

            # 3) 파일명 구성
            file_name = os.path.basename(file)
            base_name = os.path.splitext(file_name)[0]

            # 4) 알파 마스크 & bbox 크롭
            mask = output_image.getchannel("A")
            bbox = mask.getbbox()
            if bbox is None:
                print(f"[{idx}] bbox 없음 (옷 영역 인식 실패): {file}")
                continue

            output_image = output_image.crop(bbox)
            mask = mask.crop(bbox)

            # 저장: 배경제거 결과
            nobg_path = os.path.join(output_folder, f"{base_name}_nobg.png")
            output_image.save(nobg_path)
            print(f"[{idx}/{len(files)}] 완료: {nobg_path}")

            # 5) 마스크 이진화 (0/255)
            mask_array = np.array(mask)
            mask_bin = (mask_array > 0).astype(np.uint8) * 255

            # 6) 윤곽 시각화 저장
            mask_vis = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_vis, contours, -1, (0, 0, 255), 2)  # 빨간 윤곽선
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), mask_vis)

            # 7) 거리 변환(한 번만 계산) → 크롭 후보를 빠르게 샘플링
            #    distanceTransform은 각 픽셀이 '가장 가까운 0(배경) 픽셀'까지의 거리(L2)를 반환.
            #    마스크 내부(=255) 픽셀에서 값이 큼 → 내부 중심부일수록 크다.
            dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3).astype(np.float32)

            # 8) 랜덤 크롭 배치
            w, h = output_image.size
            placed_rects = []
            vis_image = output_image.copy()
            draw = ImageDraw.Draw(vis_image)

            for i in range(CROP_COUNT_PER_IMAGE):
                success = False

                # 큰 박스부터 배치되도록, 첫 시도에는 큰 비율을 우선 주는 트릭 (선택)
                first_try_ratio = random.uniform((CROP_RATIO_MIN + CROP_RATIO_MAX) / 2, CROP_RATIO_MAX)

                for attempt in range(MAX_TRIES_PER_CROP):
                    crop_ratio = (
                        first_try_ratio if attempt == 0
                        else random.uniform(CROP_RATIO_MIN, CROP_RATIO_MAX)
                    )
                    crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
                    if crop_w <= 0 or crop_h <= 0 or crop_w >= w or crop_h >= h:
                        continue

                    # 직사각형이 마스크 안에 완전히 들어가려면
                    # 중심점의 distance 값이 '사각형 대각선의 절반' 이상이어야 보수적으로 안전.
                    radius = 0.5 * math.hypot(crop_w, crop_h)

                    # 여유 간격을 마스크 경계에도 반영하고 싶다면 다음 줄의 주석 해제:
                    # radius += NON_OVERLAP_GAP_PX / 2.0

                    # 조건을 만족하는 중심 좌표 후보 (dist >= radius)
                    ys, xs = np.where(dist >= radius)
                    if len(xs) == 0:
                        continue

                    # 무작위 중심 선택
                    idx_choice = random.randint(0, len(xs) - 1)
                    cx, cy = int(xs[idx_choice]), int(ys[idx_choice])

                    left = cx - crop_w // 2
                    top = cy - crop_h // 2
                    right = left + crop_w
                    bottom = top + crop_h

                    # 이미지 경계 체크
                    if left < 0 or top < 0 or right > w or bottom > h:
                        continue

                    rect = (left, top, right, bottom)

                    # 다른 박스와 겹치지 않게 체크
                    if any(rect_intersects(rect, r, NON_OVERLAP_GAP_PX) for r in placed_rects):
                        continue

                    # 통과 → 저장
                    cropped = output_image.crop(rect)
                    crop_save_path = os.path.join(output_folder, f"{base_name}_crop{i+1}.png")
                    cropped.save(crop_save_path)

                    # 시각화(초록 박스)
                    draw.rectangle(rect, outline="lime", width=3)

                    placed_rects.append(rect)
                    print(f"   └ 랜덤 크롭 저장: {crop_save_path}")
                    success = True
                    break

                if not success:
                    print(f"   └ [경고] {i+1}번째 크롭 실패 (적합 위치 없음)")

            # 9) 시각화 이미지 저장
            vis_path = os.path.join(output_folder, f"{base_name}_visual.png")
            vis_image.save(vis_path)

        except Exception as e:
            print(f"[{idx}/{len(files)}] 실패: {file}, 이유: {e}")

    print("✅ 전체 완료 (GPU 가용 시 rembg 가속 + Distance Transform 기반 Non-overlap 크롭 + 시각화 + 마스크 저장)!")


if __name__ == "__main__":
    main()
