# -*- coding: utf-8 -*-
"""
[대형 직물 생성 - 믹스매치(Mix & Match) 모드]
- 폴더 A (경사) + 폴더 B (위사) 조합으로 랜덤 샘플 생성
- 두 개의 텍스처를 각각 생성하여 블렌더로 전달
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

# 1. 루트 폴더 (이 안의 하위 폴더들을 랜덤 조합)
ROOT_INPUT_DIR = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no4_Scanning\1117_max_packing"

# 2. 결과 저장 경로
BASE_OUTPUT_DIR = r"C:\Users\_idal\PycharmProjects\Cloth_AI\Result_Large_Weave_Mix"

# 3. 생성할 랜덤 샘플 개수
NUM_SAMPLES = 10

# 4. 설정
BLENDER_SCRIPT = "Mix_Blender.py"
SWATCH_RES_PX = 376
RANDOM_SEED = 42  # 랜덤 조합을 고정하려면 숫자 입력, 매번 다르게 하려면 None


# ==========================================
# [기능]
# ==========================================

def calculate_optimal_grid(num_files):
    if num_files == 0: return 0, 0
    cols = int(math.sqrt(num_files))
    if cols == 0: cols = 1
    rows = num_files // cols
    return cols, rows


def create_texture_from_folder(input_dir, save_path, req_cols=None, req_rows=None):
    """
    폴더의 이미지를 스티칭하여 텍스처 생성.
    req_cols, req_rows가 주어지면 그 크기에 맞춤 (안 주어지면 자동 계산)
    """
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(valid_ext)]
    num_files = len(files)

    if num_files == 0:
        return False, 0, 0

    # 그리드 계산 (요청된 크기가 있으면 그걸 따르고, 없으면 자동 계산)
    if req_cols and req_rows:
        cols, rows = req_cols, req_rows
    else:
        cols, rows = calculate_optimal_grid(num_files)

    target_count = cols * rows

    # 캔버스 준비
    total_w = cols * SWATCH_RES_PX
    total_h = rows * SWATCH_RES_PX
    master_img = Image.new("RGBA", (total_w, total_h))

    # 랜덤 섞기
    random.shuffle(files)

    # 이미지가 부족하면 반복해서 채움
    selected_files = []
    while len(selected_files) < target_count:
        selected_files.extend(files)
    selected_files = selected_files[:target_count]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            path = selected_files[idx]
            try:
                patch = Image.open(path).convert("RGBA")
                if patch.size != (SWATCH_RES_PX, SWATCH_RES_PX):
                    patch = patch.resize((SWATCH_RES_PX, SWATCH_RES_PX), Image.LANCZOS)
                master_img.paste(patch, (c * SWATCH_RES_PX, r * SWATCH_RES_PX))
            except:
                pass
            idx += 1

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    master_img.save(save_path)
    return True, cols, rows


def find_blender_auto():
    path = shutil.which("blender")
    if path: return path
    pf = r"C:\Program Files\Blender Foundation"
    if os.path.exists(pf):
        dirs = sorted([d for d in os.listdir(pf) if "Blender" in d], reverse=True)
        for d in dirs:
            exe = os.path.join(pf, d, "blender.exe")
            if os.path.exists(exe): return exe
    return None


def run_blender_mix(warp_tex, weft_tex, cols, rows, output_path):
    blender_exe = find_blender_auto()
    if not blender_exe:
        print("   ❌ 블렌더 없음")
        return

    script_path = os.path.join(os.getcwd(), BLENDER_SCRIPT)

    cmd = [
        blender_exe, "-b", "-P", script_path,
        "--",
        warp_tex,  # 1. 경사 텍스처
        weft_tex,  # 2. 위사 텍스처
        str(cols),
        str(rows),
        output_path
    ]
    subprocess.run(cmd)


def main():
    if not os.path.exists(ROOT_INPUT_DIR):
        print(f"❌ 경로 없음: {ROOT_INPUT_DIR}")
        return

    # 폴더 리스트 확보
    all_folders = [
        os.path.join(ROOT_INPUT_DIR, d)
        for d in os.listdir(ROOT_INPUT_DIR)
        if os.path.isdir(os.path.join(ROOT_INPUT_DIR, d))
    ]

    if len(all_folders) < 2:
        print("❌ 폴더가 최소 2개 이상 있어야 섞을 수 있습니다.")
        return

    print(f"========== 믹스매치 생성 시작 (총 {NUM_SAMPLES}개 예정) ==========")

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    for i in range(NUM_SAMPLES):
        # 1. 랜덤으로 두 폴더 선택 (중복 허용 여부는 choice vs sample. 여기선 서로 다른 폴더 sample)
        folder_warp, folder_weft = random.sample(all_folders, 2)

        name_warp = os.path.basename(folder_warp)
        name_weft = os.path.basename(folder_weft)

        mix_name = f"Mix_{i + 1:02d}_W({name_warp})_x_H({name_weft})"
        print(f"\n[{i + 1}/{NUM_SAMPLES}] 조합 생성: {mix_name}")

        out_dir = os.path.join(BASE_OUTPUT_DIR, mix_name)
        os.makedirs(out_dir, exist_ok=True)

        # 2. 텍스처 생성
        # 기준 크기는 Warp 폴더의 최적 크기를 따름 (물리적 크기 기준점)
        tex_warp_path = os.path.join(out_dir, "texture_warp.png")
        success_w, cols, rows = create_texture_from_folder(folder_warp, tex_warp_path)

        if not success_w:
            print("   (경사 이미지 생성 실패 - 건너뜀)")
            continue

        # 위사 텍스처는 경사 텍스처의 크기(cols, rows)에 맞춰서 생성 (강제 맞춤)
        tex_weft_path = os.path.join(out_dir, "texture_weft.png")
        success_h, _, _ = create_texture_from_folder(folder_weft, tex_weft_path, req_cols=cols, req_rows=rows)

        if not success_h:
            print("   (위사 이미지 생성 실패 - 건너뜀)")
            continue

        # 3. 블렌더 실행
        render_out = os.path.join(out_dir, f"{mix_name}.png")
        print(f"   ▶ 블렌더 렌더링... (Grid: {cols}x{rows})")
        run_blender_mix(tex_warp_path, tex_weft_path, cols, rows, render_out)
        print("   ✅ 완료")

    print("\n========== 모든 작업 완료 ==========")


if __name__ == "__main__":
    main()