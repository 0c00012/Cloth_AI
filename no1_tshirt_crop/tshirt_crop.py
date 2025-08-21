"""
Random crops strictly INSIDE a garment mask. (NO argparse version)
- 1) Try SAM AutoMaskGenerator (robust to backgrounds & garment types)
- 2) Fallback to Saliency + GrabCut if SAM isn't available or fails
- 3) Pick random rectangles guaranteed fully inside the mask (morphological test)
Saves: binary mask, overlay preview, and N random crops.

Dependencies:
  pip install opencv-python pillow numpy
  # (Optional, recommended) for SAM:
  pip install torch torchvision
  pip install git+https://github.com/facebookresearch/segment-anything.git

Configure paths below and run:  python script.py
"""

import os
import cv2
import sys
import math
import json
import random
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple  # <-- Python 3.8+ compatible typing

# =========================
# USER SETTINGS (edit here)
# =========================
INPUTS = [r"C:\Users\_idal\PycharmProjects\Cloth_AI\data\0819_data"]
OUTPUT_DIR = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt"

# Cropping / geometry
CROPS_PER_IMAGE = 5
AREA_FRAC_RANGE = (0.06, 0.20)  # relative to MASK area
ASPECT_RANGE    = (0.7, 1.4)    # w/h
BORDER_MARGIN_PX = 4

# SAM (optional)
USE_SAM   = True
SAM_CKPT  = r"C:\Users\_idal\Downloads\sam_vit_h_4b8939.pth"  # "" to disable
SAM_MODEL = "vit_h"  # vit_h | vit_l | vit_b
# =========================


# ----------------------------
# SAM (optional) lazy import
# ----------------------------
def load_sam(ckpt_path: str, model_type: str = "vit_h"):
    try:
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        if not ckpt_path or not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        sam.to(device)
        auto = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            box_nms_thresh=0.6,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=512,   # remove tiny
        )
        print(f"[INFO] SAM loaded ({model_type}) on {device}")
        return auto
    except Exception as e:
        print(f"[WARN] SAM not available or failed to load: {e}")
        return None

# ----------------------------
# Mask utilities
# ----------------------------
def biggest_component(mask: np.ndarray) -> np.ndarray:
    num, lbl, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num <= 1:
        return mask
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lbl == idx).astype(np.uint8) * 255

def close_fill(mask: np.ndarray, k_close=13) -> np.ndarray:
    if k_close > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8), iterations=1)
    # fill holes
    h, w = mask.shape
    flood = mask.copy()
    ffmask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, ffmask, (0,0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inv)

def mask_from_sam(auto_gen, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Generate a single garment-like mask from SAM AutoMaskGenerator outputs."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    proposals = auto_gen.generate(image_rgb)
    if not proposals:
        return None

    H, W = image_bgr.shape[:2]
    cx, cy = W/2, H/2
    img_area = H * W

    # filter & score: prefer large-enough masks near the center; penalize extreme aspect ratios
    scored = []
    for p in proposals:
        m = p["segmentation"].astype(np.uint8)
        area = m.sum()
        if area < 0.03*img_area or area > 0.85*img_area:
            continue
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        mx, my = xs.mean(), ys.mean()
        d = math.hypot(mx - cx, my - cy) / math.hypot(cx, cy)  # 0~1
        y1, x1, y2, x2 = ys.min(), xs.min(), ys.max(), xs.max()
        ar = (x2-x1+1) / (y2-y1+1 + 1e-6)
        skinny_penalty = min(ar, 1/ar)
        score = (area/img_area) * (1.0 - d) * skinny_penalty
        scored.append((score, m*255))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    mask = scored[0][1]
    mask = biggest_component(mask)
    mask = close_fill(mask, 15)
    return mask

def mask_from_grabcut(image_bgr: np.ndarray) -> np.ndarray:
    """Fallback: saliency -> initial rect -> GrabCut -> binary mask."""
    H, W = image_bgr.shape[:2]
    # 1) saliency to get rough bbox (OpenCV fine-grained)
    try:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()
        ok, salmap = sal.computeSaliency(image_bgr)
        if not ok:
            raise RuntimeError("saliency compute failed")
    except Exception:
        # Some OpenCV builds don't have saliency module
        salmap = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        salmap = cv2.GaussianBlur(salmap, (0,0), 3)
        salmap = cv2.normalize(salmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    thr = (salmap > (salmap.mean() + salmap.std())).astype(np.uint8) * 255
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((9,9), np.uint8), 1)
    thr = biggest_component(thr)
    ys, xs = np.where(thr > 0)
    if len(xs) < 10:  # degenerate
        x1, y1, x2, y2 = int(W*0.1), int(H*0.1), int(W*0.9), int(H*0.9)
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

    # 2) GrabCut
    rect = (max(0, x1-10), max(0, y1-10), min(W-1, x2-x1+20), min(H-1, y2-y1+20))
    gc_mask = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    cv2.grabCut(image_bgr, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    binmask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    binmask = biggest_component(binmask)
    binmask = close_fill(binmask, 11)
    return binmask

# ----------------------------
# Random crop strictly inside mask
# ----------------------------
def random_rect_inside_mask(
    mask: np.ndarray,
    area_frac_range: Tuple[float, float] = (0.06, 0.20),
    aspect_range: Tuple[float, float] = (0.7, 1.4),
    border_margin_px: int = 4,
    max_tries: int = 40
) -> Optional[Tuple[int, int, int, int]]:
    """Return (x1,y1,x2,y2) fully inside mask using erosion feasibility."""
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    m[:, :border_margin_px] = 0
    m[:, -border_margin_px:] = 0
    m[:border_margin_px, :] = 0
    m[-border_margin_px:, :] = 0

    m_area = int(m.sum())
    if m_area < 256:
        return None

    for _ in range(max_tries):
        target = random.uniform(*area_frac_range) * m_area
        aspect = random.uniform(*aspect_range)   # w/h
        h = int(max(16, min(H-2, math.sqrt(target / max(aspect,1e-6)))))
        w = int(max(16, min(W-2, int(h * aspect))))

        kernel = np.ones((h, w), np.uint8)
        feas = cv2.erode(m, kernel, iterations=1)
        ys, xs = np.where(feas > 0)
        if len(xs) == 0:
            continue
        k = np.random.randint(0, len(xs))
        cy, cx = int(ys[k]), int(xs[k])
        x1, y1 = int(cx - w//2), int(cy - h//2)
        x2, y2 = x1 + w, y1 + h

        sub = m[y1:y2, x1:x2]
        if sub.size > 0 and sub.sum() == sub.shape[0] * sub.shape[1]:
            return (x1, y1, x2, y2)
    return None

# ----------------------------
# Core processing
# ----------------------------
def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def gather_images(inputs) -> List[Path]:
    """Accept list of files/folders; return sorted unique image paths (recursive for folders)."""
    out: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for ext in (".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"):
                out.extend(p.rglob(f"*{ext}"))
        elif p.exists() and is_image(p):
            out.append(p)
        else:
            print(f"[WARN] Skipping (not found or not an image): {item}")
    # unique + sorted
    return sorted({q.resolve() for q in out})

def process_image(
    img_path: Path,
    out_dir: Path,
    auto_sam,
    crops_per_image: int = 1,
    area_frac_range: Tuple[float, float] = (0.06, 0.20),
    aspect_range: Tuple[float, float] = (0.7, 1.4),
    border_margin_px: int = 4
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"[ERR] Failed to read: {img_path}")
        return

    # 1) get mask
    mask: Optional[np.ndarray] = None
    if auto_sam is not None:
        try:
            mask = mask_from_sam(auto_sam, bgr)
        except Exception as e:
            print(f"[WARN] SAM failed on {img_path.name}: {e}")
            mask = None

    if mask is None:
        mask = mask_from_grabcut(bgr)

    # 2) save mask
    stem = img_path.stem
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    # 3) crops
    saved = []
    for i in range(crops_per_image):
        rect = random_rect_inside_mask(
            mask,
            area_frac_range=area_frac_range,
            aspect_range=aspect_range,
            border_margin_px=border_margin_px
        )
        if rect is None:
            print(f"[WARN] No feasible crop for {img_path.name} (try relaxing ranges).")
            break
        x1, y1, x2, y2 = rect
        crop = bgr[y1:y2, x1:x2].copy()
        crop_path = out_dir / f"{stem}_crop{i+1}.png"
        cv2.imwrite(str(crop_path), crop)
        saved.append({"crop_idx": i+1, "rect": [int(x1),int(y1),int(x2),int(y2)], "path": str(crop_path)})

    # 4) overlay preview
    overlay = bgr.copy()
    contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0,0,255), 3)
    for s in saved:
        x1,y1,x2,y2 = s["rect"]
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 3)
    over_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(over_path), overlay)

    meta = {
        "image": str(img_path),
        "mask": str(mask_path),
        "overlay": str(over_path),
        "crops": saved
    }
    with open(out_dir / f"{stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] {img_path.name}: mask→{mask_path.name}, crops={len(saved)}, overlay→{over_path.name}")

def run() -> None:
    out_dir = Path(OUTPUT_DIR)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    auto_sam = load_sam(SAM_CKPT, SAM_MODEL) if (USE_SAM and SAM_CKPT) else None
    imgs = gather_images(INPUTS)
    if not imgs:
        print("[ERR] No images found from INPUTS. Check your paths.")
        return

    print(f"[INFO] Found {len(imgs)} image(s). Output: {out_dir}")
    for p in imgs:
        process_image(
            p, out_dir, auto_sam,
            crops_per_image=CROPS_PER_IMAGE,
            area_frac_range=AREA_FRAC_RANGE,
            aspect_range=ASPECT_RANGE,
            border_margin_px=BORDER_MARGIN_PX
        )

if __name__ == "__main__":
    run()
