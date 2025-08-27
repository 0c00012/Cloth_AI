# Fabric megavariations generator (≈20 variants)
# deps: pillow, numpy
from PIL import Image, ImageFilter, ImageOps
import numpy as np, os, math, itertools, pathlib, random

# ------------ Paths ------------
src_path   = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no2_blender_swatch_maker\render_output\2img\img_2_tshirt.png"  # <<-- 여기에 너의 이미지 경로
output_dir = r".\fabric_variations"                       # 저장 폴더

os.makedirs(output_dir, exist_ok=True)

# ------------ Utils ------------
def to_np(img):  return np.asarray(img).astype(np.float32)/255.0
def to_img(a):   return Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8))
def normalize01(a):
    m, M = float(a.min()), float(a.max())
    return np.zeros_like(a) if M-m<1e-6 else (a-m)/(M-m)

def soft_light(base, blend):
    return (1 - 2*blend) * (base**2) + 2 * blend * base

def overlay(base, blend):
    return np.where(base<=0.5, 2*base*blend, 1-2*(1-base)*(1-blend))

def multiply(a,b): return np.clip(a*b,0,1)

def rgb(r,g,b):    return np.array([r,g,b], dtype=np.float32)
def fill_like(gray, col):  # gray(H,W) -> (H,W,3) tint
    return np.dstack([gray*col[0], gray*col[1], gray*col[2]])

# ------------ Height/Weave primitives ------------
def rounded_profile(dist, radius, softness=0.25):
    # 1 at center -> 0 to edges, smooth round
    core = np.clip(1 - (dist/(radius+1e-6))**2, 0, 1)
    t = 1 - np.clip((dist - (radius*(1-softness))) / (radius*softness + 1e-6), 0, 1)
    return np.maximum(core, t*core)

def make_uv(H,W,tile):
    yy, xx = np.mgrid[0:H,0:W]
    u = xx % tile; v = yy % tile
    return xx, yy, u, v

def thread_fields(H,W,tile,thread_ratio=0.62,edge_soft=0.25):
    _, _, u, v = make_uv(H,W,tile)
    radius = thread_ratio*tile/2.0
    du = np.abs(u - tile/2)
    dv = np.abs(v - tile/2)
    V = rounded_profile(du, radius, edge_soft)  # vertical thread profile
    Hh = rounded_profile(dv, radius, edge_soft) # horizontal thread profile
    return V, Hh

def shading_from_height(h, gloss_intensity=0.0, spec_power=20,
                        light_dir=(0.7,0.6,0.25), view_dir=(0,0,1)):
    # normals (finite differences)
    gx = (np.roll(h,-1,1) - np.roll(h,1,1))*0.5
    gy = (np.roll(h,-1,0) - np.roll(h,1,0))*0.5
    nz = np.full_like(h, 0.7); nx = -gx; ny = -gy
    nlen = np.sqrt(nx*nx + ny*ny + nz*nz) + 1e-6
    nx /= nlen; ny /= nlen; nz /= nlen
    L = np.array(light_dir, dtype=np.float32)
    L /= np.linalg.norm(L)+1e-6
    V = np.array(view_dir, dtype=np.float32)
    V /= np.linalg.norm(V)+1e-6
    # Lambert
    lam = normalize01(np.clip(nx*L[0]+ny*L[1]+nz*L[2], 0, 1))
    # Specular (Blinn-Phong)
    Hh = (L+V); Hh /= np.linalg.norm(Hh)+1e-6
    ndoth = np.clip(nx*Hh[0]+ny*Hh[1]+nz*Hh[2], 0, 1)
    spec = np.power(ndoth, max(1,spec_power))
    # 0.78~1.18 범위
    return np.clip(0.78 + 0.40*lam + gloss_intensity*spec*0.25, 0.6, 1.35)

# ------------ Pattern masks ------------
def overmask_plain(H,W,tile):
    yy, xx, *_ = make_uv(H,W,tile)
    return ((xx//tile + yy//tile) % 2)==1

def overmask_basket(H,W,tile,group=2):
    yy, xx, *_ = make_uv(H,W,tile)
    return (((xx//tile)//group + (yy//tile)//group) % 2)==1

def overmask_twill(H,W,tile,over=2,under=1,offset=1):
    # vertical over when (x//tile + offset*(y//tile)) % period < over
    yy, xx, *_ = make_uv(H,W,tile)
    period = over+under
    return ((xx//tile + (yy//tile)*offset) % period) < over

def overmask_herringbone(H,W,tile,over=2,under=1,block_rows=6):
    yy, xx, *_ = make_uv(H,W,tile)
    period = over+under
    blk = ((yy//tile)//block_rows) % 2
    off = np.where(blk==0, 1, -1)
    return ((xx//tile + (yy//tile)*off) % period) < over

def overmask_satin(H,W,tile,harness=5,step=2,over_count=None):
    # simple satin: over when (i*step + j) % harness < over_count
    yy, xx, *_ = make_uv(H,W,tile)
    if over_count is None: over_count = harness-1  # e.g., 5h satin ~ 4/1
    return (((xx//tile)*step + (yy//tile)) % harness) < over_count

# ------------ Linen (non-interlaced) ------------
def linen_texture(H,W,seed=42,strength=0.45):
    rng = np.random.default_rng(seed)
    # vertical and horizontal streaks + fine fiber grain
    col = rng.normal(0,1,size=W)
    row = rng.normal(0,1,size=H)
    k = np.array([1,2,3,4,3,2,1], np.float32); k/=k.sum()
    col = np.convolve(col,k,mode="same")
    row = np.convolve(row,k,mode="same")
    col_tex = np.tile(col[np.newaxis,:], (H,1))
    row_tex = np.tile(row[:,np.newaxis], (1,W))
    base = normalize01(col_tex*0.8 + row_tex*0.8)
    fine = rng.normal(0,1,size=(H,W))
    fine = (fine + np.roll(fine,1,0)+np.roll(fine,-1,0)+np.roll(fine,1,1)+np.roll(fine,-1,1))/5.0
    fine = normalize01(fine)
    tex = normalize01(base*0.75 + fine*0.25)
    return tex

# ------------ Compositing ------------
def apply_ink_bleed(base_rgb, bleed_radius=0.0, bleed_strength=0.0, bg_assume_white=True):
    if bleed_radius<=0 or bleed_strength<=0: return base_rgb
    H,W,_ = base_rgb.shape
    # ink mask ~ how far from white (assuming art on light cloth)
    gray = (0.299*base_rgb[:,:,0] + 0.587*base_rgb[:,:,1] + 0.114*base_rgb[:,:,2])
    ink_mask = 1 - gray if bg_assume_white else gray
    mask_img = to_img(ink_mask)
    mask_blur = to_np(mask_img.filter(ImageFilter.GaussianBlur(radius=bleed_radius)).convert("L"))
    # blur colors locally
    img = to_img(base_rgb)
    blur_rgb = to_np(img.filter(ImageFilter.GaussianBlur(radius=max(1,bleed_radius*0.8))))
    amount = np.clip(mask_blur*bleed_strength, 0, 1)[:,:,None]
    return np.clip(base_rgb*(1-amount) + blur_rgb*amount, 0, 1)

def weave_render(im_rgb, pattern, tile=18, thread_ratio=0.62, edge_soft=0.25,
                 gloss=0.0, spec_power=20, ink_bleed=0.0, bleed_strength=0.0,
                 print_mode="surface", warp_col=rgb(0.98,0.98,0.98), weft_col=rgb(0.98,0.98,0.98),
                 extra=None):
    H,W = im_rgb.shape[:2]
    base = im_rgb.copy()

    # --- patterns ---ukhuhuhiu
    if pattern=="linen":
        tex = linen_texture(H,W, seed=extra.get("seed",42) if extra else 42)
        emb = to_np(to_img(tex).filter(ImageFilter.Kernel((3,3),[-2,-1,0,-1,1,1,0,1,2],1)).convert("L"))
        # soft-light the texture
        out = base*0.55 + soft_light(base, np.dstack([tex]*3))*0.45
        out = np.clip(out*(0.95 + emb[:,:,None]*0.10), 0, 1)
        out = apply_ink_bleed(out, bleed_radius=ink_bleed, bleed_strength=bleed_strength)
        return out

    # woven patterns
    Vt, Ht = thread_fields(H,W,tile,thread_ratio,edge_soft)
    if pattern=="plain":
        over = overmask_plain(H,W,tile)
    elif pattern=="basket2":
        over = overmask_basket(H,W,tile,group=2)
    elif pattern=="basket3":
        over = overmask_basket(H,W,tile,group=3)
    elif pattern=="twill21":
        over = overmask_twill(H,W,tile,over=2,under=1,offset=1)
    elif pattern=="twill31":
        over = overmask_twill(H,W,tile,over=3,under=1,offset=1)
    elif pattern=="herringbone":
        over = overmask_herringbone(H,W,tile,over=2,under=1,block_rows=extra.get("block_rows",6) if extra else 6)
    elif pattern=="satin5":
        over = overmask_satin(H,W,tile,harness=5,step=2,over_count=4)
    elif pattern=="satin8":
        over = overmask_satin(H,W,tile,harness=8,step=3,over_count=7)
    else:
        over = overmask_plain(H,W,tile)

    over = over.astype(np.float32)
    height = np.where(over==1, Vt*1.0 + Ht*0.25, Ht*1.0 + Vt*0.25)
    height = normalize01(height)

    shade = shading_from_height(height, gloss_intensity=gloss, spec_power=spec_power)
    # thread tint (warp/weft)
    warp = fill_like(Vt, warp_col); weft = fill_like(Ht, weft_col)
    tint = normalize01(warp + weft)  # off-white default

    # printing model
    printed = apply_ink_bleed(base, bleed_radius=ink_bleed, bleed_strength=bleed_strength)
    if print_mode=="yarn":
        # yarn-dyed background color + print overlay
        yarn = np.clip(tint*(0.86 + 0.20*height[:,:,None]), 0, 1)
        printed = np.clip(overlay(yarn, printed), 0, 1)
    # shading
    shaded = np.clip(printed * shade[:,:,None], 0, 1)
    # soft light with tint to pick up fibers
    out = np.clip(soft_light(shaded, tint), 0, 1)
    return out

# ------------ Load ------------
im = Image.open(src_path).convert("RGBA")
W,H = im.size
im_rgb = to_np(im.convert("RGB"))

# ------------ Variation specs (≈20) ------------
# pattern: linen/plain/basket2/basket3/twill21/twill31/herringbone/satin5/satin8
# tile: 조밀(8~14), 중간(16~22), 성글(24~32)
# gloss: 0(매트)~1(글로시), ink_bleed(px), bleed_strength(0~1)
# print_mode: "surface" | "yarn"
VARIATIONS = [
    dict(name="01_linen_tight_matte",     pattern="linen",   tile=0,  gloss=0.0, spec_power=16, ink_bleed=0.0, bleed_strength=0.0, extra={"seed":21}),
    dict(name="02_linen_coarse_soft",     pattern="linen",   tile=0,  gloss=0.1, spec_power=20, ink_bleed=1.5, bleed_strength=0.25, extra={"seed":77}),
    dict(name="03_plain_tight_matte",     pattern="plain",   tile=12, gloss=0.0, spec_power=18, ink_bleed=0.8, bleed_strength=0.20),
    dict(name="04_plain_loose_matte",     pattern="plain",   tile=26, gloss=0.0, spec_power=18, ink_bleed=0.8, bleed_strength=0.20),
    dict(name="05_plain_tight_gloss",     pattern="plain",   tile=12, gloss=0.6, spec_power=60, ink_bleed=0.5, bleed_strength=0.15),
    dict(name="06_canvas_heavy",          pattern="plain",   tile=22, gloss=0.1, spec_power=24, ink_bleed=1.0, bleed_strength=0.25,
         ),  # 캔버스 느낌(굵은 평직)
    dict(name="07_basket2_matte",         pattern="basket2", tile=18, gloss=0.0, spec_power=18, ink_bleed=0.6, bleed_strength=0.18),
    dict(name="08_basket2_gloss",         pattern="basket2", tile=18, gloss=0.5, spec_power=50, ink_bleed=0.4, bleed_strength=0.12),
    dict(name="09_basket3_loose",         pattern="basket3", tile=24, gloss=0.2, spec_power=28, ink_bleed=0.7, bleed_strength=0.20),
    dict(name="10_twill21_tight_matte",   pattern="twill21", tile=14, gloss=0.0, spec_power=18, ink_bleed=0.6, bleed_strength=0.18),
    dict(name="11_twill21_gloss",         pattern="twill21", tile=14, gloss=0.55,spec_power=60, ink_bleed=0.5, bleed_strength=0.16),
    dict(name="12_twill31_loose",         pattern="twill31", tile=22, gloss=0.15,spec_power=26, ink_bleed=1.2, bleed_strength=0.28),
    dict(name="13_herringbone_tight",     pattern="herringbone", tile=14, gloss=0.1, spec_power=24, ink_bleed=0.8, bleed_strength=0.22, extra={"block_rows":5}),
    dict(name="14_herringbone_loose_gloss",pattern="herringbone", tile=24, gloss=0.5, spec_power=50, ink_bleed=0.6, bleed_strength=0.18, extra={"block_rows":7}),
    dict(name="15_satin5_glossy",         pattern="satin5",  tile=16, gloss=0.8, spec_power=90, ink_bleed=0.3, bleed_strength=0.10),
    dict(name="16_satin8_highgloss",      pattern="satin8",  tile=18, gloss=1.0, spec_power=120,ink_bleed=0.2, bleed_strength=0.10),
    dict(name="17_denim_matte_surface",   pattern="twill31", tile=18, gloss=0.05,spec_power=24, ink_bleed=1.0, bleed_strength=0.27,
         print_mode="surface",
         # 인디고 워프/화이트 웨프트
         warp_col=rgb(0.08,0.15,0.30), weft_col=rgb(0.92,0.92,0.95)),
    dict(name="18_denim_yarn_dyed",       pattern="twill31", tile=18, gloss=0.10,spec_power=30, ink_bleed=0.6, bleed_strength=0.20,
         print_mode="yarn",
         warp_col=rgb(0.10,0.17,0.35), weft_col=rgb(0.95,0.95,0.98)),
    dict(name="19_chambray_light",        pattern="plain",   tile=16, gloss=0.10,spec_power=24, ink_bleed=0.8, bleed_strength=0.22,
         print_mode="yarn",
         warp_col=rgb(0.45,0.60,0.90), weft_col=rgb(0.96,0.96,0.98)),
    dict(name="20_vintage_bleed_heavy",   pattern="twill21", tile=20, gloss=0.0, spec_power=18, ink_bleed=2.4, bleed_strength=0.45),
    dict(name="21_plain_satiny_coated",   pattern="plain",   tile=16, gloss=0.75,spec_power=80, ink_bleed=0.3, bleed_strength=0.10),
]

# ------------ Run all variations ------------
def render_and_save(spec):
    p = dict(pattern="plain", tile=18, thread_ratio=0.62, edge_soft=0.25,
             gloss=0.0, spec_power=20, ink_bleed=0.0, bleed_strength=0.0,
             print_mode="surface", warp_col=rgb(0.98,0.98,0.98), weft_col=rgb(0.98,0.98,0.98),
             extra=None)
    p.update(spec)
    pattern = p.pop("pattern"); name = p.pop("name")
    out = weave_render(im_rgb, pattern=pattern, **p)

    rgba = to_img(out).convert("RGBA"); rgba.putalpha(im.split()[-1])
    save_path = os.path.join(output_dir, f"{name}.png")
    rgba.save(save_path)
    print("Saved:", save_path)

for spec in VARIATIONS:
    render_and_save(spec)

print("\nDone. Variations saved to:", os.path.abspath(output_dir))
