# -*- coding: utf-8 -*-
"""
Shared constants and folder layout for the paper pipeline.

Paper correspondence (draft 0909):
  3.3  Garment segmentation      -> 1_segment_garments.py
  3.4  Swatch extraction         -> 2_extract_swatches.py   (grid-origin optimization)
  3.5  Texture synthesis         -> 3_synthesize_master_texture.py
  3.6  Yarn-based 3D weaving     -> 4_render_weave.py + Blender_Large_Weave.py
  4.4  Placement comparison      -> experiments/compare_placement_strategies.py
  legacy/                        -> raster-scan greedy baseline and older scripts
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- folders (all relative to Final_pipeline/) --------------------------------
INPUT_IMAGE_DIR = ROOT / "data" / "input_images"        # original garment photos
SEGMENT_DIR = ROOT / "work" / "01_segmented"            # *_nobg.png (RGBA), *_mask.png
SWATCH_DIR = ROOT / "work" / "02_swatches"              # <garment>/<garment>_crop_NN.png
TEXTURE_DIR = ROOT / "work" / "03_master_textures"      # <garment>_master_texture_CxR.png
RENDER_DIR = ROOT / "outputs" / "04_weave_renders"      # Blender Cycles renders
BLENDER_SCRIPT = ROOT / "Blender_Large_Weave.py"

# ---- physical scale mapping (paper 3.4.1) ---------------------------------------
# The 100 cm wide capture area is normalised to 3,755 px  ->  37.55 px/cm.
REF_WIDTH_PX = 3755
REF_WIDTH_CM = 100.0
PX_PER_CM = REF_WIDTH_PX / REF_WIDTH_CM          # 37.55 px/cm
SWATCH_CM = 10.0                                  # nominal swatch size 10 x 10 cm
SWATCH_PX = int(round(SWATCH_CM * PX_PER_CM))     # 375.5 -> 376 px

# ---- segmentation mask rule (paper 3.3.2) ---------------------------------------
ALPHA_THRESHOLD = 10                              # alpha > 10  -> garment foreground
REMBG_MODEL = "u2net"                             # pretrained U^2-Net, no fine-tuning

# ---- texture synthesis (paper 3.5.2) --------------------------------------------
SHUFFLE_SEED = 42                                 # fixed seed for swatch order randomisation

# ---- sample naming used in the paper --------------------------------------------
GARMENT_IDS = {
    "all-over pattern_rotate": "G01",
    "check_1_rotate": "G02",
    "print_1_rotate": "G03",
    "stripe_1_rotate": "G04",
    "stripe_2_rotate": "G05",
}


def garment_id(stem: str) -> str:
    """Map a file stem such as 'check_1_rotate' to its paper label (G02)."""
    stem = stem.replace("_nobg", "")
    return GARMENT_IDS.get(stem, stem)
