import os
from PIL import Image
from tqdm import tqdm  # 진행상황 표시용 (선택사항, pip install tqdm)


def crop_width_1px(input_dir):
    # 1. 저장할 경로 생성 (원본 보호를 위해 '_cropped' 폴더 생성)
    output_dir = input_dir + "_cropped"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory created: {output_dir}")
    else:
        print(f"Saving to existing directory: {output_dir}")

    # 2. 이미지 확장자 정의
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # 3. 파일 리스트 가져오기
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

    if not files:
        print("No image files found in the directory.")
        return

    print(f"Processing {len(files)} images...")

    # 4. 이미지 처리 루프
    for filename in tqdm(files):
        try:
            file_path = os.path.join(input_dir, filename)

            with Image.open(file_path) as img:
                width, height = img.size

                # 논리적 유효성 검사: 너비가 1픽셀보다 커야 자를 수 있음
                if width <= 1:
                    print(f"[Skip] Image too narrow: {filename}")
                    continue

                # --- 크롭 영역 설정 (Left, Upper, Right, Lower) ---
                # 가로(너비)를 1픽셀 줄임 (오른쪽 끝을 1px 잘라냄)
                # 좌표는 (0, 0)에서 시작하여 (width-1, height)까지 자름
                crop_area = (0, 0, width - 1, height)

                # 만약 '왼쪽'을 1픽셀 자르고 싶다면 아래 주석을 해제하여 사용:
                # crop_area = (1, 0, width, height)

                # 만약 '양쪽'을 1픽셀씩(총 2픽셀) 자르고 싶다면:
                # crop_area = (1, 0, width - 1, height)

                cropped_img = img.crop(crop_area)

                # 저장 (파일명 유지)
                save_path = os.path.join(output_dir, filename)
                cropped_img.save(save_path)

        except Exception as e:
            print(f"[Error] Failed to process {filename}: {e}")

    print("All tasks completed.")


if __name__ == "__main__":
    # Windows 경로 문자열 (r prefix 사용)
    target_path = r"C:\Users\_idal\PycharmProjects\Cloth_AI\renders_1117\combo2"

    if os.path.exists(target_path):
        crop_width_1px(target_path)
    else:
        print(f"Path does not exist: {target_path}")