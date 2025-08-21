"""
Garment-aware random crops STRICTLY inside mask (no argparse).
Upgrades:
- Pose-guided ROI -> SAM box-prompt for robust masks (tank-top, sleeveless, short sleeves, pants)
- Optional mild skin suppression
- Non-overlapping crops with optional gap
Fallbacks: SAM Auto -> GrabCut.
"""

import os, cv2, sys, math, json, random, numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

# =========================
# USER SETTINGS (edit here)
# =========================
INPUTS = [r"C:\Users\_idal\PycharmProjects\Cloth_AI\data\0819_data"]
OUTPUT_DIR = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\0819_cropped_tshirt"

# Cropping / geometry
CROPS_PER_IMAGE   = 5
AREA_FRAC_RANGE   = (0.06, 0.20)  # relative to MASK area
ASPECT_RANGE      = (0.7, 1.4)    # w/h
BORDER_MARGIN_PX  = 4
NON_OVERLAP_GAP_PX = 0            # set 6~12 for visible gaps

# Masking guidance
GUIDE_WITH_POSE   = True          # use MediaPipe Pose to guide SAM with a box prompt
TARGET_GARMENT    = "upper"       # "upper" or "lower"
ROI_EXPAND        = (0.12, 0.08)  # (x_frac, y_frac) expand around pose box
REMOVE_SKIN       = False         # try True if 팔/목이 자주 붙는 경우

# SAM (optional)
USE_SAM   = True
SAM_CKPT  = r"C:\Users\_idal\Downloads\sam_vit_h_4b8939.pth"
SAM_MODEL = "vit_h"  # vit_h | vit_l | vit_b
# =========================

# ----------------------------
# SAM (optional) loader
# ----------------------------
def load_sam(ckpt_path: str, model_type: str = "vit_h"):
    try:
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
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
            min_mask_region_area=512,
        )
        predictor = SamPredictor(sam)
        print(f"[INFO] SAM loaded ({model_type}) on {device}")
        return {"auto": auto, "predictor": predictor, "device": device}
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
    h, w = mask.shape
    flood = mask.copy()
    ffmask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(flood, ffmask, (0,0), 255)
    inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inv)

def skin_mask_ycrcb(image_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    # Classic skin box (broad). Tune if needed.
    skin = cv2.inRange(ycrcb, (0,133,77), (255,173,127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), 1)
    skin = cv2.morphologyEx(skin, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), 1)
    return skin

# ----------------------------
# Pose-guided ROI (MediaPipe)
# ----------------------------
def upper_lower_box_via_pose(image_bgr: np.ndarray, target: str = "upper",
                             expand_xy: Tuple[float, float] = (0.12, 0.08)) -> Optional[np.ndarray]:
    try:
        import mediapipe as mp
        H, W = image_bgr.shape[:2]
        pose = mp.solutions.pose.Pose(static_image_mode=True, enable_segmentation=False)
        res = pose.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        pose.close()
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks.landmark

        def pix(i):
            return int(lm[i].x * W), int(lm[i].y * H)

        # indices: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
        LS, RS, LH, RH = 11, 12, 23, 24
        xs = []
        ys_top = []
        ys_bot = []
        for i in (LS, RS, LH, RH):
            x, y = pix(i)
            xs.append(x)
            if i in (LS, RS): ys_top.append(y)
            else: ys_bot.append(y)

        x1, x2 = min(xs), max(xs)
        y_sh_top = min(ys_top)
        y_hip_bot = max(ys_bot)

        if target == "upper":
            y1, y2 = int(y_sh_top - 0.20*H), int(y_hip_bot + 0.08*H)
        else:  # lower
            # from hips downward
            y1, y2 = int(y_hip_bot - 0.05*H), int(min(H-1, y_hip_bot + 0.55*H))

        # expand
        ex, ey = expand_xy
        dx, dy = int(ex * W), int(ey * H)
        x1, y1 = max(0, x1 - dx), max(0, y1 - dy)
        x2, y2 = min(W-1, x2 + dx), min(H-1, y2 + dy)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        return np.array([x1, y1, x2, y2], dtype=np.int32)
    except Exception as e:
        print(f"[WARN] Pose guide failed: {e}")
        return None

# ----------------------------
# SAM-based masking
# ----------------------------
def mask_from_sam_box(predictor, image_bgr: np.ndarray, box_xyxy: np.ndarray,
                      remove_skin: bool=False) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    masks, scores, _ = predictor.predict(box=box_xyxy[None, :], multimask_output=True)
    if masks is None or len(masks) == 0:
        return None
    idx = int(np.argmax(scores))
    m = (masks[idx].astype(np.uint8) * 255)
    m = biggest_component(m)
    if remove_skin:
        skin = skin_mask_ycrcb(image_bgr)
        m = cv2.bitwise_and(m, cv2.bitwise_not(skin))
    m = close_fill(m, 11)
    return m

def mask_from_sam_auto(auto_gen, image_bgr: np.ndarray) -> Optional[np.ndarray]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    proposals = auto_gen.generate(image_rgb)
    if not proposals:
        return None
    H, W = image_bgr.shape[:2]
    cx, cy = W/2, H/2
    img_area = H * W
    scored = []
    for p in proposals:
        m = p["segmentation"].astype(np.uint8)
        area = m.sum()
        if area < 0.03*img_area or area > 0.90*img_area:
            continue
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            continue
        mx, my = xs.mean(), ys.mean()
        d = math.hypot(mx - cx, my - cy) / math.hypot(cx, cy)
        y1, x1, y2, x2 = ys.min(), xs.min(), ys.max(), xs.max()
        ar = (x2-x1+1) / (y2-y1+1 + 1e-6)
        skinny_penalty = min(ar, 1/ar)
        scored.append(((area/img_area) * (1.0 - d) * skinny_penalty, (m*255)))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    mask = scored[0][1]
    return close_fill(biggest_component(mask), 15)

def mask_from_grabcut(image_bgr: np.ndarray, guide_box: Optional[np.ndarray]=None) -> np.ndarray:
    H, W = image_bgr.shape[:2]
    if guide_box is None:
        try:
            sal = cv2.saliency.StaticSaliencyFineGrained_create()
            ok, salmap = sal.computeSaliency(image_bgr)
            if not ok: raise RuntimeError()
        except Exception:
            salmap = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            salmap = cv2.GaussianBlur(salmap, (0,0), 3)
            salmap = cv2.normalize(salmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thr = (salmap > (salmap.mean() + salmap.std())).astype(np.uint8) * 255
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((9,9), np.uint8), 1)
        thr = biggest_component(thr)
        ys, xs = np.where(thr > 0)
        if len(xs) < 10:
            guide_box = np.array([int(W*0.1), int(H*0.1), int(W*0.9), int(H*0.9)], dtype=np.int32)
        else:
            guide_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.int32)

    x1,y1,x2,y2 = guide_box.tolist()
    rect = (max(0, x1-10), max(0, y1-10), min(W-1, x2-x1+20), min(H-1, y2-y1+20))
    gc_mask = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    cv2.grabCut(image_bgr, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    binmask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return close_fill(biggest_component(binmask), 11)

# ----------------------------
# Rect helpers (non-overlap)
# ----------------------------
def rect_intersects(a, b, gap=0) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    bx1g, by1g, bx2g, by2g = bx1 - gap, by1 - gap, bx2 + gap, by2 + gap
    if ax2 <= bx1g or bx2g <= ax1 or ay2 <= by1g or by2g <= ay1:
        return False
    return True

# ----------------------------
# Random crop strictly inside mask
# ----------------------------
def random_rect_inside_mask(
    mask: np.ndarray,
    area_frac_range: Tuple[float, float] = (0.06, 0.20),
    aspect_range: Tuple[float, float] = (0.7, 1.4),
    border_margin_px: int = 4,
    max_tries: int = 80,
    avoid: Optional[List[Tuple[int,int,int,int]]] = None,
    non_overlap_gap_px: int = 0
) -> Optional[Tuple[int, int, int, int]]:
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    m[:, :border_margin_px] = 0; m[:, -border_margin_px:] = 0
    m[:border_margin_px, :] = 0; m[-border_margin_px:, :] = 0
    m_area = int(m.sum())
    if m_area < 256: return None
    avoid = avoid or []
    for _ in range(max_tries):
        target = random.uniform(*area_frac_range) * m_area
        aspect = random.uniform(*aspect_range)
        h = int(max(16, min(H-2, math.sqrt(target / max(aspect,1e-6)))))
        w = int(max(16, min(W-2, int(h * aspect))))
        kernel = np.ones((h, w), np.uint8)
        feas = cv2.erode(m, kernel, iterations=1)
        ys, xs = np.where(feas > 0)
        if len(xs) == 0: continue
        k = np.random.randint(0, len(xs))
        cy, cx = int(ys[k]), int(xs[k])
        x1, y1 = int(cx - w//2), int(cy - h//2)
        x2, y2 = x1 + w, y1 + h
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H: continue
        rect = (x1, y1, x2, y2)
        if any(rect_intersects(rect, r, gap=non_overlap_gap_px) for r in avoid): continue
        sub = m[y1:y2, x1:x2]
        if sub.size > 0 and sub.sum() == sub.shape[0] * sub.shape[1]:
            return rect
    return None

# ----------------------------
# Core processing
# ----------------------------
def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def gather_images(inputs) -> List[Path]:
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
    return sorted({q.resolve() for q in out})

def obtain_mask(bgr: np.ndarray, sam_bundle, target_garment: str) -> np.ndarray:
    H, W = bgr.shape[:2]
    guide_box = None
    if GUIDE_WITH_POSE:
        guide_box = upper_lower_box_via_pose(bgr, target=target_garment, expand_xy=ROI_EXPAND)

    # 1) SAM with box prompt (best)
    if sam_bundle is not None and guide_box is not None:
        m = mask_from_sam_box(sam_bundle["predictor"], bgr, guide_box, remove_skin=REMOVE_SKIN)
        if m is not None: return m

    # 2) SAM auto (next)
    if sam_bundle is not None:
        m = mask_from_sam_auto(sam_bundle["auto"], bgr)
        if m is not None:
            if REMOVE_SKIN:
                skin = skin_mask_ycrcb(bgr)
                m = cv2.bitwise_and(m, cv2.bitwise_not(skin))
                m = close_fill(m, 9)
            return m

    # 3) GrabCut (fallback), prefer guided rect if we have it
    return mask_from_grabcut(bgr, guide_box=guide_box)

def process_image(
    img_path: Path,
    out_dir: Path,
    sam_bundle,
    crops_per_image: int = 1,
    area_frac_range: Tuple[float, float] = (0.06, 0.20),
    aspect_range: Tuple[float, float] = (0.7, 1.4),
    border_margin_px: int = 4,
    non_overlap_gap_px: int = 0
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"[ERR] Failed to read: {img_path}"); return

    # 1) get mask (pose-guided -> auto -> grabcut)
    mask = obtain_mask(bgr, sam_bundle if (USE_SAM and SAM_CKPT) else None, TARGET_GARMENT)

    # 2) save mask
    stem = img_path.stem
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)

    # 3) crops (non-overlapping)
    placed_rects, saved = [], []
    for i in range(crops_per_image):
        rect = random_rect_inside_mask(
            mask,
            area_frac_range=area_frac_range,
            aspect_range=aspect_range,
            border_margin_px=border_margin_px,
            max_tries=100,
            avoid=placed_rects,
            non_overlap_gap_px=non_overlap_gap_px
        )
        if rect is None:
            print(f"[WARN] No feasible NON-OVERLAPPING crop for {img_path.name} at i={i+1}.")
            break
        x1, y1, x2, y2 = rect
        crop = bgr[y1:y2, x1:x2].copy()
        crop_path = out_dir / f"{stem}_crop{i+1}.png"
        cv2.imwrite(str(crop_path), crop)
        placed_rects.append(rect)
        saved.append({"crop_idx": i+1, "rect": [int(x1),int(y1),int(x2),int(y2)], "path": str(crop_path)})

    # 4) overlay preview
    overlay = bgr.copy()
    contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0,0,255), 3)
    for s in saved:
        x1,y1,x2,y2 = s["rect"]
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
    over_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(over_path), overlay)

    meta = {"image": str(img_path), "mask": str(mask_path), "overlay": str(over_path), "crops": saved}
    with open(out_dir / f"{stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[OK] {img_path.name}: mask→{mask_path.name}, crops={len(saved)}, overlay→{over_path.name}]")

def run() -> None:
    out_dir = Path(OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    sam_bundle = load_sam(SAM_CKPT, SAM_MODEL) if (USE_SAM and SAM_CKPT) else None
    imgs = gather_images(INPUTS)
    if not imgs:
        print("[ERR] No images found from INPUTS. Check your paths."); return
    print(f"[INFO] Found {len(imgs)} image(s). Output: {out_dir}")
    for p in imgs:
        process_image(
            p, out_dir, sam_bundle,
            crops_per_image=CROPS_PER_IMAGE,
            area_frac_range=AREA_FRAC_RANGE,
            aspect_range=ASPECT_RANGE,
            border_margin_px=BORDER_MARGIN_PX,
            non_overlap_gap_px=NON_OVERLAP_GAP_PX
        )

if __name__ == "__main__":
    run()
