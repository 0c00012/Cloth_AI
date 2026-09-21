# -*- coding: utf-8 -*-
"""
[대형 직물 생성 파이프라인 - 자동 폴더 탐색 모드]
- 지정된 부모 폴더(ROOT_INPUT_DIR) 내의 모든 하위 폴더를 자동으로 검색합니다.
- 각 폴더를 순회하며 텍스처를 생성하고 블렌더 렌더링을 수행합니다.
"""

import os
import math
import random
import subprocess
import shutil
from PIL import Image

# ==========================================
# [사용자 설정 CONFIG]
# ==========================================

# 1. ★ 스와치 폴더들이 들어있는 '최상위 부모 폴더' 경로
# (calc_max_area_and_stitch.py의 OUTPUT_DIR 경로를 넣으세요)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_INPUT_DIR = os.path.join(SCRIPT_DIR, "work", "max_packing")

# 2. 결과가 저장될 폴더
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "large_weave")

# 3. 블렌더 스크립트 파일명
BLENDER_SCRIPT = os.path.join(SCRIPT_DIR, "Blender_Large_Weave.py")

# 4. 물리적 기준 (10cm = 376px)
SWATCH_RES_PX = 376

# 5. 랜덤 시드 고정 (None이면 매번 다름)
RANDOM_SEED = 42


# ==========================================
# [기능]
# ==========================================

def calculate_optimal_grid(num_files):
    """
    파일 개수(N)에 맞춰 가장 효율적인 가로x세로 비율을 계산합니다.
    """
    if num_files == 0:
        return 0, 0

    cols = int(math.sqrt(num_files))
    if cols == 0: cols = 1
    rows = num_files // cols
    return cols, rows


def create_master_texture_dynamic(input_dir, save_path):
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]

    num_files = len(files)
    if num_files == 0:
        # 이미지 없는 폴더는 조용히 패스하거나 로그만 남김
        return False, 0, 0

    print(f"   ▶ 원본 스와치: {num_files}개 발견")

    # 자동 그리드 계산
    cols, rows = calculate_optimal_grid(num_files)
    target_count = cols * rows

    if target_count == 0:
        print("   ❌ 유효한 격자를 만들 수 없습니다.")
        return False, 0, 0

    print(f"   ▶ 최적 그리드: {cols} x {rows} (총 {target_count}장 사용)")

    # 캔버스 크기 계산
    total_w = cols * SWATCH_RES_PX
    total_h = rows * SWATCH_RES_PX

    master_img = Image.new("RGBA", (total_w, total_h))

    # 랜덤 시드 적용
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # 랜덤 섞기 후 중복 없이 순서대로 사용
    random.shuffle(files)
    selected_files = files[:target_count]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(selected_files):
                break

            choice_path = selected_files[idx]
            try:
                patch = Image.open(choice_path).convert("RGBA")
                if patch.size != (SWATCH_RES_PX, SWATCH_RES_PX):
                    patch = patch.resize((SWATCH_RES_PX, SWATCH_RES_PX), Image.LANCZOS)

                x = c * SWATCH_RES_PX
                y = r * SWATCH_RES_PX
                master_img.paste(patch, (x, y))
            except Exception as e:
                print(f"      [Error] {choice_path}: {e}")

            idx += 1

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    master_img.save(save_path)
    print(f"   ✅ 텍스처 생성 완료: {os.path.basename(save_path)}")

    return True, cols, rows


def find_blender_auto():
    env_path = os.environ.get("BLENDER_EXE")
    if env_path and os.path.exists(env_path):
        return env_path

    path = shutil.which("blender")
    if path: return path
    pf = r"C:\Program Files\Blender Foundation"
    if os.path.exists(pf):
        dirs = sorted([d for d in os.listdir(pf) if "Blender" in d], reverse=True)
        for d in dirs:
            exe = os.path.join(pf, d, "blender.exe")
            if os.path.exists(exe): return exe
    return None


def run_blender(texture_path, cols, rows, output_folder, folder_name):
    blender_exe = find_blender_auto()
    if not blender_exe:
        print("   ❌ 블렌더 실행 파일을 찾을 수 없습니다.")
        return

    script_path = BLENDER_SCRIPT

    # 렌더링 파일명 설정 (Result/폴더명/폴더명_4x5.png)
    render_filename = f"{folder_name}_{cols}x{rows}.png"
    render_out_path = os.path.join(output_folder, render_filename)

    print(f"   ▶ 블렌더 렌더링 시작... -> {render_filename}")

    cmd = [
        blender_exe, "-b", "-P", script_path,
        "--",
        texture_path,
        str(cols),
        str(rows),
        render_out_path
    ]

    subprocess.run(cmd)
    print(f"   ✅ 렌더링 완료")


def main():
    if not os.path.exists(ROOT_INPUT_DIR):
        print(f"❌ 루트 입력 경로가 존재하지 않습니다: {ROOT_INPUT_DIR}")
        return

    # ★ 자동 탐색 로직: ROOT 폴더 안의 모든 '폴더'만 찾아서 리스트로 만듦
    subfolders = [
        os.path.join(ROOT_INPUT_DIR, d)
        for d in os.listdir(ROOT_INPUT_DIR)
        if os.path.isdir(os.path.join(ROOT_INPUT_DIR, d))
    ]

    print(f"========== 일괄 처리 시작 (총 {len(subfolders)}개 폴더 발견) ==========")
    print(f"📂 루트 경로: {ROOT_INPUT_DIR}")

    count = 0
    for input_dir in subfolders:
        folder_name = os.path.basename(input_dir)

        # (선택) 특정 폴더 제외하고 싶으면 여기에 조건 추가 (예: if folder_name == "__pycache__": continue)

        print(f"\n[{count + 1}/{len(subfolders)}] 탐색 중: {folder_name}")

        # 1. 개별 아웃풋 폴더 생성
        specific_output_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
        os.makedirs(specific_output_dir, exist_ok=True)

        # 2. 임시 텍스처 경로
        temp_tex_path = os.path.join(specific_output_dir, "temp_texture.png")

        # 3. 텍스처 생성 (이미지가 없으면 False 반환됨)
        success, final_cols, final_rows = create_master_texture_dynamic(input_dir, temp_tex_path)

        # 4. 블렌더 실행 (성공 시에만)
        if success:
            run_blender(temp_tex_path, final_cols, final_rows, specific_output_dir, folder_name)
            count += 1
        else:
            print("   (이미지가 없거나 부족하여 건너뜀)")

    print("\n========== 모든 작업이 완료되었습니다 ==========")


if __name__ == "__main__":
    main()
