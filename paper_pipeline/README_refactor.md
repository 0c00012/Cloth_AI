# Cloth AI Refactored Pipeline

## 구성 파일

* `cloth\_pipeline\_config.py`  
공통 스케일, threshold, seed, swatch 크기를 한 곳에서 관리합니다.
* `preprocess\_segment\_refactored.py`  
배경 제거 + 의류 영역 크롭을 수행합니다.
* `calc\_max\_area\_and\_stitch\_refactored.py`  
정확한 square-fit 판정과 multi-offset greedy packing으로 swatch를 추출합니다.
* `split\_swatches\_10h10v.py`  
필요할 때만 swatch를 10h / 10v로 분할합니다.
* `Run\_Large\_Weave\_refactored.py`  
모든 swatch를 사용해 2D texture를 만들고 Blender를 호출합니다.
* `Blender\_Large\_Weave\_refactored.py`  
실제 목표 크기에 맞게 thread pitch를 재보정해 3D weave를 렌더링합니다.

## 권장 실행 순서

### 1\) segmentation

```bash
python preprocess\_segment\_refactored.py \\
  --input-dir "<원본 이미지 폴더>" \\
  --output-dir "<segmentation 결과 폴더>"
```

### 2\) swatch packing

```bash
python calc\_max\_area\_and\_stitch\_refactored.py \\
  --input-dir "<segmentation 결과 폴더>" \\
  --output-dir "<packing 결과 폴더>" \\
  --scan-step-px 10 \\
  --offset-mode coarse
```

### 3\) optional 10h/10v slicing

```bash
python split\_swatches\_10h10v.py \\
  --input-dir "<packing 결과 폴더>" \\
  --output-dir "<slice 결과 폴더>"
```

### 4\) 2D texture + Blender render

```bash
python Run\_Large\_Weave\_refactored.py \\
  --root-input-dir "<packing 결과 폴더>" \\
  --base-output-dir "<최종 결과 폴더>" \\
  --blender-script "Blender\_Large\_Weave\_refactored.py"
```

## 리팩토링 핵심 포인트

1. **전처리 역할 분리**  
배경 제거 단계와 swatch/slice 단계를 분리했습니다.
2. **정확한 swatch 포함 판정**  
distance transform 대신 integral image로 `10cm x 10cm` 정사각형이 마스크 내부에 완전히 들어가는지 검사합니다.
3. **경계 누락 보정**  
오른쪽/아래쪽 끝 좌표도 반드시 검사하도록 off-by-one 문제를 수정했습니다.
4. **모든 swatch 사용**  
texture 조립 시 남는 swatch를 버리지 않고, 빈 칸이 있더라도 전체 swatch를 모두 사용합니다.
5. **물리 단위 일관성 개선**  
Blender 단계에서 목표 크기에 맞춰 유효 pitch를 다시 계산해 실제 렌더 크기와 논문 상 10cm grid가 어긋나지 않도록 했습니다.
6. **재현성 / 추적성 강화**  
CSV/JSON/log 파일을 저장하고, 정렬된 파일 순서와 고정 seed를 사용합니다.

## 참고 산출물

* segmentation 단계: `segmentation\_summary.csv`
* packing 단계: `packing\_summary.csv`, 폴더별 `placements.csv`
* rendering 단계: `render\_summary.csv`, 폴더별 `\*\_layout.json`, `blender\_render.log`

