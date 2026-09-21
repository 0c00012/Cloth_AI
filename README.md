# Cloth_AI — 폐의류 이미지의 물리적 스케일 기반 디지털 원단화 및 원사 기반 3D 직조 시각화

폐의류 사진 한 장에서 **의류 영역 분할 → 10 × 10 cm 스와치 추출 → 마스터 텍스처 합성 → Blender 원사 기반 직조 렌더링**까지 이어지는
파이프라인입니다. 논문 초안(2026-09)의 방법·결과와 1:1로 대응하는 코드는 **`Final_pipeline/`** 하나이며, 나머지 폴더는 이전 실험 기록입니다.

## 파이프라인 (`Final_pipeline/`)

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 1 | `1_segment_garments.py` | 사전학습 U²-Net(rembg)으로 배경 제거, alpha > 10 이진 마스크 |
| 2 | `2_extract_swatches.py` | 3,755 px = 100 cm 스케일에서 376 px(10 cm) 정규 격자를 141,376개 시작 위치로 평행이동해 마스크 안에 완전히 포함되는 셀 수가 최대인 격자를 선택 (grid-origin optimization) |
| 3 | `3_synthesize_master_texture.py` | floor 격자(cols = ⌊√N⌋, rows = N // cols), seed 42 셔플로 마스터 텍스처 합성 |
| 4 | `4_render_weave.py` + `Blender_Large_Weave.py` | 경사/위사 Curve, 직사각형 bevel, crimp, Cycles 렌더링 |
| 비교 | `experiments/compare_placement_strategies.py` | raster 그리디 · 고정 격자 · 무작위 · 최적 격자 비교 (논문 Table 3, Figure 9–10) |
| 기준선 | `legacy/greedy_raster_packing.py` | 이전 raster-scan 그리디 (최종 방법이 아님) |

다섯 벌(G01–G05)의 논문 결과: 추출 21 / 26 / 32 / 21 / 22장(합계 122), 합성 사용 115장(94.3 %), 격자 4×5 / 5×5 / 5×6 / 4×5 / 4×5.
격자 시작 위치·셀 좌표·합성 구성은 `Final_pipeline/reference_results/`에 JSON/CSV로 들어 있습니다.
설치·실행 방법은 [Final_pipeline/README.md](Final_pipeline/README.md)를 보세요.

```powershell
cd Final_pipeline
python 1_segment_garments.py
python 2_extract_swatches.py
python 3_synthesize_master_texture.py
python 4_render_weave.py        # Blender 4.5 필요
```

## 저장소 구성

```text
Final_pipeline/          논문 파이프라인 (위 표)
  data/input_images/     원본 의류 사진 5장 (3,755 × 2,628 px)
  reference_results/     논문 수치 재현용 배치·합성 결과 (JSON/CSV)
  legacy/, experiments/, mix/
0327_AI/                 (이전) 스와치 생성 LoRA 실험
0428_dino_guided_swatch_pipeline/  (이전) DINO 유도 스와치 확장 실험
fabricdiffusion_actual_pipeline/   (이전) FabricDiffusion 기반 실험
no0_size_calc/ … no4_Scanning/     (이전) 단계별 초기 스크립트
최대수율_과정.py           (이전) 그리디 과정 시각화
```

`work/`, `outputs/`, 렌더 결과, 대용량 데이터·zip은 `.gitignore`로 제외되어 있으며 스크립트로 다시 만들 수 있습니다.
