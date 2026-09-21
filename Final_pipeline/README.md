# Cloth AI – paper pipeline (Final_pipeline)

폐의류 이미지 → U²-Net 의류 분할 → **정규 격자 시작 위치 최적화(grid-origin optimization)** 스와치 추출 →
마스터 텍스처 합성 → Blender 원사 기반 직조 렌더링. 논문 초안(draft 0915) 3장의 단계와 1:1로 대응합니다.

| 단계 | 스크립트 | 논문 절 | 입력 → 출력 |
|---|---|---|---|
| 1 | `1_segment_garments.py` | 3.3 | `data/input_images/*.jpg` → `work/01_segmented/<stem>_nobg.png`, `_mask.png` |
| 2 | `2_extract_swatches.py` | 3.4 | `work/01_segmented/*_nobg.png` → `work/02_swatches/<stem>/<stem>_crop_NN.png`, `_placement.json`, `_layout.png` |
| 3 | `3_synthesize_master_texture.py` | 3.5 | `work/02_swatches/` → `work/03_master_textures/<stem>_master_texture_CxR.png` |
| 4 | `4_render_weave.py` (+ `Blender_Large_Weave.py`) | 3.6 | `work/03_master_textures/` → `outputs/04_weave_renders/<stem>/<stem>_CxR.png` |
| 보조 | `render_crimp_closeup.py` | 3.6.3 (Figure 5) | 스와치 1장 → 10 × 10 cm 한 셀의 크림프 확대 렌더 2장 |
| 비교 | `experiments/compare_placement_strategies.py` | 3.4.6 / 4.4 | 그리디·고정 격자·무작위·최적 격자 비교 (Table 3, Figure 9–10) |
| 예비 | `experiments/grid_rotation_search.py` | 4.4 | 격자 방향(0–85°) + 시작 위치 탐색 예비 실험 (5종 합계 122 → 123) |
| 전체 | `run_all.py` | — | 1→4단계 일괄 실행 (`--skip-segment`, `--skip-render`) |
| 테스트 | `tests/test_extract_swatches.py` | — | `pytest tests` (적분영상·격자 집계·동률 해소·극대성 검증) |
| 기준선 | `legacy/greedy_raster_packing.py` | 4.4의 "raster greedy" | 이전 방식(10 px stride 그리디). 최종 방법이 아님 |

공통 상수는 `pipeline_config.py`에 있습니다: 3,755 px = 100 cm(37.55 px/cm), 스와치 376 px = 10 cm,
전경 규칙 alpha > 10, 텍스처 합성 seed 42, 샘플 ID(G01–G05) 매핑.

## 설치

```powershell
cd Final_pipeline
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Blender 4.5가 필요합니다. `BLENDER_EXE` 환경변수, PATH의 `blender`, 또는
`C:\Program Files\Blender Foundation\Blender *\blender.exe` 순으로 자동 탐색합니다.
`1_segment_garments.py`는 처음 실행 시 `rembg`의 `u2net` 가중치를 내려받습니다(인터넷 필요).

## 실행

```powershell
python 1_segment_garments.py          # U²-Net 배경 제거 + 이진 마스크
python 2_extract_swatches.py          # 141,376개 격자 시작 위치 전수 비교 → 스와치 저장
python 3_synthesize_master_texture.py # floor 격자(cols=⌊√N⌋, rows=N//cols), seed 42
python 4_render_weave.py              # Blender Cycles 렌더링 (GPU 자동)
```

논문 결과 재현: `work/01_segmented/`에는 논문에 사용한 원본 세그멘테이션 결과 5장이 들어 있습니다
(원본 프로젝트 `no1_tshirt_crop/1110_2_crop_size_1010/*_nobg.png`와 동일). 이 입력으로 2단계를 실행하면
G01–G05 = 21, 26, 32, 21, 22개(합계 122), 격자 시작 위치 (27,154), (230,244), (1,237), (40,331), (124,89) px가 나옵니다(여유 기준 동률 해소; 이전 규칙에서는 (34,147), (227,231), (346,208), (40,331), (118,66)).
3단계 결과는 20, 25, 30, 20, 20개 사용(합계 115, 94.3 %), 격자 4×5, 5×5, 5×6, 4×5, 4×5입니다.
1단계를 다시 실행하면 rembg 버전에 따라 마스크가 미세하게 달라질 수 있으므로, 논문 수치 재현에는 제공된 파일을 그대로 쓰세요.

## 추출 방법 요약 (`2_extract_swatches.py`)

1. alpha > 10 → 이진 의류 마스크.
2. 적분 영상으로 모든 좌상단 위치에 대해 "376 × 376 px 창이 마스크 안에 완전히 포함되는지"를 한 번에 계산.
3. 셀 크기 = 간격 = 376 px인 정규 격자를 (ox, oy) ∈ [0, 375]² 만큼 평행이동한 141,376가지 격자에 대해
   유효 셀 수를 집계(좌표를 376 주기로 그룹화하는 reshape-sum).
4. 유효 셀 수가 최대인 시작 위치를 선택. 동률(5종에서 1~8,088개)이면 셀들의 마스크 경계까지 최소 거리(cell margin)가 가장 큰 격자 → 평균 여유 → 작은 y → 작은 x 순으로 선택(`--no-tie-break`로 이전 규칙 사용).
5. 선택된 격자의 유효 셀을 행 우선 순서로 저장하고, 포함·비중첩·면적을 픽셀 단위로 재검증. 잔여 영역에 376 px 정사각형이 더 들어갈 수 있는지(극대성)와 면적 상한 ⌊전경/376²⌋도 placement.json에 기록.

이 값은 "평행이동한 정규 격자 계열 안에서의 최댓값"이며, 자유 배치 전체의 전역 최댓값을 뜻하지 않습니다.

## 폴더

```text
Final_pipeline/
  pipeline_config.py, 1_… 4_… .py, Blender_Large_Weave.py, render_crimp_closeup.py
  data/input_images/           원본 이미지 5장
  work/01_segmented/           세그멘테이션 결과 (논문 입력 포함)
  work/02_swatches/            스와치, placement.json, layout.png, grid_origin_counts.npy
  work/03_master_textures/     마스터 텍스처 + composition.json + texture_composition_summary.csv
  work/max_packing/            (이전) 그리디 결과 사본 — mix/ 실험 입력으로만 사용
  outputs/04_weave_renders/    Blender 렌더링 결과
  outputs/crimp_closeup/       Figure 5용 크림프 확대 렌더
  experiments/                 배치 전략 비교 스크립트 (결과 폴더는 실행 시 생성)
  legacy/                      이전 스크립트 (greedy_raster_packing.py, preprocess_crop_random_swatches.py, Run_Large_Weave.py)
  mix/                         두 의류의 스와치를 섞는 별도 실험 분기 (논문 본문에는 미포함)
```

Blender 세부 설정(원사 간격 0.5/0.8 cm, 단면 0.35×0.15 / 0.65×0.20 cm, crimp 0.32 × 0.7, 위사 Z +0.3 mm,
Cycles 2048 samples, adaptive 0.01, OptiX denoise, area light 22 W × 셀 수, orthographic camera 등)은
`Blender_Large_Weave.py` 상단 상수와 논문 Appendix A(Table A1–A4)에 같은 값으로 정리되어 있습니다.
