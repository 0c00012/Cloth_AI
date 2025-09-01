import os
from glob import glob
from rembg import remove
from PIL import Image

# 입력 폴더와 출력 폴더 경로
input_folder = r"C:\Users\IDAL\Desktop\KakaoTalk_20230321_093803765_18"
output_folder = r"C:\Users\IDAL\Desktop\cloth_ai"

# 출력 폴더 없으면 생성




os.makedirs(output_folder, exist_ok=True)

# 이미지 확장자 리스트
exts = ['*.jpg', '*.jpeg', '*.png']

# 모든 이미지 파일 불러오기
files = []
for ext in exts:
    files.extend(glob(os.path.join(input_folder, ext)))

print(f"총 {len(files)}개 이미지 처리 시작...")

for idx, file in enumerate(files, 1):
    try:
        # 이미지 열기
        input_image = Image.open(file).convert("RGB")

        # 배경 제거
        output_image = remove(input_image)

        # 파일 이름만 추출
        file_name = os.path.basename(file)

        # 출력 경로 (확장자는 png로 저장)
        save_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + "_nobg.png")

        # 결과 저장
        output_image.save(save_path)

        print(f"[{idx}/{len(files)}] 완료: {save_path}")
    except Exception as e:
        print(f"[{idx}/{len(files)}] 실패: {file}, 이유: {e}")

print("✅ 전체 배경 제거 완료!")
