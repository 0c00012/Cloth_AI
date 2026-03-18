# -*- coding: utf-8 -*-
"""
Weave swatch generator (single / 2-combo / 3-combo)
- 카메라, 조명(World+Light_Top), UV, 머티리얼 마스크(윗면만 보이기) 등은 1110 스크립트 로직을 그대로 따름
- 경사(세로실)=흰색 고정 + 폭은 위사 대비 0.7배(더 얇게)
- 각 실 사이 간격을 약간 줄이기: 실 폭을 전국적으로 키우는 팩터(WIDTH_TIGHTEN_FACTOR)
- 폴더 자동 인식: *_H_00.png..H_09.png을 가진 폴더 탐색, base name으로 그룹핑 (ex: check_1_rotate_crop_rot1 → base: check_1)
- 콤보 출력 강화: base별 여러 variant(rot1..5, crop1..5)를 동시에 조합하여 combo2/3 더 많이 생성
- 상세 로그로 경로/로딩 확인 및 디버깅 플래그 제공 (하얗게 나올 때 원인 파악)

USAGE (Blender):
  blender -b -P weave_swatch_combo_1110_v2.py

필요시 CONFIG 섹션만 수정.
"""

import bpy, re, os
from pathlib import Path
from itertools import combinations, product
from math import sin, pi

# =========================
# CONFIG
# =========================
# Root containing band folders OR a single band folder itself
BAND_INPUT_PATH = r"C:\\Users\\_idal\\PycharmProjects\\Cloth_AI\\no1_tshirt_crop\\1110_2_crop_size_1010\\slices_10h_10v"

# Output root
OUTPUT_DIR       = r"C:\\Users\\_idal\\PycharmProjects\\Cloth_AI\\renders_1117"
OUTPUT_BASENAME  = ""  # optional prefix; if non-empty, used as <prefix>_<basename>_<ts>.png
OUTPUT_ALPHA     = True
PNG_BIT_DEPTH    = '8'        # '8' or '16'
PNG_COMPRESSION  = 15         # 0..100

# Swatch geometry (10 cm × 10 cm, pitch 1 cm)
SWATCH_SIZE_CM = 10.0
CELL_PITCH_CM  = 1.0          # 절대 바꾸지 않음 (H_00~09 = 정확히 10줄 유지)

# --- Gap tightening: 간격을 살짝 줄이기 (실 폭 증가)
WEFT_WIDTH_RATIO     = 0.45   # 기본 위사 폭 비율(피치 대비)
WARP_WIDTH_SCALE     = 0.55   # 경사 폭 = 위사 폭 * 0.70 (얇게)
WIDTH_TIGHTEN_FACTOR = 1.12   # 1.0보다 크면 전체 실 폭을 키워 간격이 줄어듦 (권장: 1.05~1.15)
MAX_WIDTH_RATIO_CAP  = 0.95   # 안전 상한 (피치의 95%)
THICK_RATIO          = 0.25   # 두께(경/위사 동일)

# Renderer & camera
USE_CYCLES  = True
SAMPLES     = 128
RESOLUTION  = (1500, 1500)
CM_EXPOSURE = 0.05

# World & single area light (Light_Top)
WORLD_STRENGTH     = 0.15
WORLD_COLOR        = (1.0, 1.0, 1.0, 1.0)
LIGHT_SIZE_MULT    = 1.25
LIGHT_TOP_ENERGY_W = 22.0
LIGHT_TOP_HEIGHT_M = 1.5

# Crimp control
CRIMP_RATIO = 0.32
CRIMP_SCALE = 0.7
CRIMP_ABS_CM = None        # e.g., 0.03 for 0.3 mm absolute amplitude

# Curve → mesh details
AUTO_CLEARANCE = False
CLEAR_MARGIN   = 0.05
STEPS_PER_CELL = 6
BEVEL_RES      = 3
RESOLUTION_U   = 24

# Z-fighting fix
WEFT_Z_OFFSET_M = 0.0003

# Colors
WARP_WHITE      = (1,1,1,1)
WEFT_SIDE_COLOR = (0,0,0,0)

# H files order flag
H_FILES_ARE_TOP_TO_BOTTOM = True

# What to render
RENDER_SINGLE_BY_FOLDER = True
RENDER_COMBO2_AUTO      = True
RENDER_COMBO3_AUTO      = True

# 디버그: 윗면 마스크 무시하고 이미지 강제 적용(하얗게 출력될 때 원인 확인용)
DEBUG_FORCE_IMAGE_ON_TOP = False

# ==== 콤보 확장 옵션 ("좀 더 뽑기") ====
# base별 variant(폴더) 선택 전략: 'best' | 'firstN' | 'all'
COMBO_VARIANT_PICK_MODE = 'firstN'
VARIANTS_PER_BASE       = 3   # 'firstN'일 때 base당 최대 3개 variant 사용 (rot1..5, 그 다음 crop1..5 우선)
# 안전 상한 (조합 과다 방지)
MAX_COMBO2_PER_PAIR     = 9   # 각 base쌍 당 최대 9장 (=3×3)
MAX_COMBO3_PER_TRIPLE   = 8   # 각 base세트 당 최대 8장 (3×3×? → 내부에서 컷)

# 옵션: 특정 콤보를 강제 추가
EXPLICIT_COMBO2 = [
    # (r"...\\all-over_pattern_rotate_crop_rot1", r"...\\check_1_rotate_crop_rot3"),
]
EXPLICIT_COMBO3 = [
    # (r"...\\all-over_pattern_rotate_crop_rot1", r"...\\check_1_rotate_crop_rot3", r"...\\print_1_rotate_crop4"),
]

# Material top-mask ramps (keep original look)
TOP_MASK_LOW  = 0.65
TOP_MASK_HIGH = 0.95

# =========================
# DERIVED
# =========================
CM2M        = 0.01
pitch_m     = CELL_PITCH_CM * CM2M
warp_cnt    = weft_cnt = int(round(SWATCH_SIZE_CM / CELL_PITCH_CM))  # =10

# 폭 계산 (간격 축소 반영)
_weft_ratio = min(MAX_WIDTH_RATIO_CAP, WEFT_WIDTH_RATIO * WIDTH_TIGHTEN_FACTOR)
weft_w      = pitch_m * _weft_ratio
warp_w      = weft_w * WARP_WIDTH_SCALE
warp_t      = weft_t = pitch_m * THICK_RATIO


def _compute_crimp_amp():
    if CRIMP_ABS_CM is not None:
        return float(CRIMP_ABS_CM) * CM2M
    return pitch_m * CRIMP_RATIO * CRIMP_SCALE

crimp_amp = _compute_crimp_amp()

# =========================
# UTIL / SCENE SETUP (kept from original scripts)
# =========================

def nuke_scene(keep_world=True):
    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')
    except: pass
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    scn=bpy.context.scene; world=scn.world if keep_world else None
    root=scn.collection
    for c in list(root.children): root.children.unlink(c)
    def purge():
        for coll in (bpy.data.objects,bpy.data.meshes,bpy.data.curves,bpy.data.cameras,
                     bpy.data.lights,bpy.data.materials,bpy.data.images,
                     bpy.data.textures,bpy.data.node_groups,bpy.data.collections):
            for db in list(coll):
                try:
                    if hasattr(db,'users') and db.users==0: coll.remove(db)
                except: pass
    for _ in range(4): purge()
    scn.world = world if (keep_world and world) else None


def set_units_cm():
    us=bpy.context.scene.unit_settings
    us.system='METRIC'
    us.length_unit='CENTIMETERS'


def enable_cycles_cuda():
    try:
        prefs=bpy.context.preferences; cp=prefs.addons['cycles'].preferences
        for dev in ('CUDA','OPTIX'):
            try:
                cp.compute_device_type=dev; cp.get_devices()
                for d in cp.devices: d.use=True
                bpy.context.scene.cycles.device='GPU'
                break
            except: pass
        try:
            bpy.context.scene.cycles.denoiser='OPTIX'
            bpy.context.view_layer.cycles.use_denoising=True
        except:
            try:
                bpy.context.scene.cycles.denoiser='OPENIMAGEDENOISE'
                bpy.context.view_layer.cycles.use_denoising=True
            except: pass
    except: pass


def setup_renderer():
    scn=bpy.context.scene
    scn.render.resolution_x, scn.render.resolution_y = RESOLUTION
    if USE_CYCLES:
        scn.render.engine='CYCLES'; scn.cycles.samples=SAMPLES
        if hasattr(scn.cycles,'use_adaptive_sampling'):
            scn.cycles.use_adaptive_sampling=True; scn.cycles.adaptive_threshold=0.01
        enable_cycles_cuda()
        try: scn.cycles.max_bounces = 4
        except: pass
        try: scn.cycles.filter_width = 0.2
        except: pass
    else:
        scn.render.engine='BLENDER_EEVEE'
    vs=scn.view_settings
    try: vs.view_transform='Filmic'
    except: pass
    vs.exposure=CM_EXPOSURE; vs.gamma=1.0


def ensure_world_background(strength: float, color=(1,1,1,1)):
    scn = bpy.context.scene
    world = scn.world or bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    bg  = None
    for n in nt.nodes:
        if getattr(n, 'bl_idname', '') == "ShaderNodeBackground":
            bg = n; break
    if bg is None:
        bg = nt.nodes.new("ShaderNodeBackground")
    out.location = (300, 0); bg.location = (0, 0)
    for link in list(nt.links):
        if link.to_node == out:
            nt.links.remove(link)
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    try: bg.inputs["Color"].default_value = color
    except: pass
    bg.inputs["Strength"].default_value = float(strength)


def setup_world_and_single_light(total_w, total_h):
    scn=bpy.context.scene
    ensure_world_background(WORLD_STRENGTH, WORLD_COLOR)
    for obj in list(scn.objects):
        if obj.type=='LIGHT' and obj.name != 'Light_Top':
            bpy.data.objects.remove(obj, do_unlink=True)
    lt=bpy.data.objects.get("Light_Top")
    if lt is None:
        ldat=bpy.data.lights.new("Light_Top", type='AREA')
        lt=bpy.data.objects.new("Light_Top", ldat)
        scn.collection.objects.link(lt)
    s=max(total_w,total_h)
    lt.data.type='AREA'; lt.data.shape='RECTANGLE'
    lt.data.size   = max(s*LIGHT_SIZE_MULT, 0.05)
    lt.data.size_y = max(s*LIGHT_SIZE_MULT, 0.05)
    lt.location = (0.0, 0.0, float(LIGHT_TOP_HEIGHT_M))
    lt.rotation_euler = (0,0,0)
    lt.data.energy = float(LIGHT_TOP_ENERGY_W)
    try: lt.data.shadow_soft_size = 0.05
    except: pass
    try: lt.data.use_contact_shadow = True
    except: pass

# =========================
# GEOMETRY
# =========================

def make_rect_profile(name, width, thickness):
    cu=bpy.data.curves.new(name=name,type='CURVE'); cu.dimensions='2D'; cu.fill_mode='BOTH'
    sp=cu.splines.new('POLY'); sp.use_cyclic_u=True
    hw,ht=float(width)*0.5,float(thickness)*0.5
    pts=[(-hw,-ht,0),(hw,-ht,0),(hw,ht,0),(-hw,ht,0)]
    sp.points.add(len(pts)-1)
    for i,(x,y,z) in enumerate(pts): sp.points[i].co=(x,y,z,1.0)
    obj=bpy.data.objects.new(name,cu); bpy.context.scene.collection.objects.link(obj)
    return obj


def poly_curve(name, points):
    cu=bpy.data.curves.new(name=name,type='CURVE'); cu.dimensions='3D'; cu.fill_mode='FULL'
    sp=cu.splines.new('POLY'); sp.points.add(len(points)-1)
    for i,(x,y,z) in enumerate(points): sp.points[i].co=(x,y,z,1.0)
    return bpy.data.objects.new(name,cu)


def build_axis_curve(axis, fixed_pos, count_other, spacing_along, idx_self, mat,
                     width, thickness, steps_per_cell=6, crimp_amp=0.0032, bevel_profile_obj=None):
    half_len=(count_other*spacing_along)*0.5
    total_steps=count_other*steps_per_cell
    pts=[]
    for s in range(total_steps+1):
        t=s/total_steps*(count_other*spacing_along)
        j=min(int(t//spacing_along), count_other-1)
        u=(t - j*spacing_along)/spacing_along
        if axis=='warp':
            x=fixed_pos; y=-half_len+t
            over=((idx_self+j)%2)==0; sign=+1 if over else -1
            z=sign*crimp_amp*sin(pi*u)
        else:
            x=-half_len+t; y=fixed_pos
            over=((j+idx_self)%2)==0; sign=-1 if over else +1
            z=sign*crimp_amp*sin(pi*u)
            z += WEFT_Z_OFFSET_M
        pts.append((x,y,z))
    obj=poly_curve(f"{'Warp' if axis=='warp' else 'Weft'}_{idx_self:02d}", pts)
    obj.data.bevel_mode='OBJECT'
    if bevel_profile_obj is not None: obj.data.bevel_object=bevel_profile_obj
    obj.data.bevel_resolution=BEVEL_RES; obj.data.resolution_u=RESOLUTION_U
    try: obj.data.twist_mode='Z_UP'
    except: pass
    if obj.data.materials: obj.data.materials[0]=mat
    else: obj.data.materials.append(mat)
    return obj


def convert_to_mesh(objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active=objs[0]
    bpy.ops.object.convert(target='MESH')
    return [bpy.data.objects.get(o.name) for o in objs]

# =========================
# UV & MATERIALS
# =========================

def ensure_uv_layer(obj, name="UVMap"):
    if obj.type!='MESH': return
    if name not in [l.name for l in obj.data.uv_layers]:
        obj.data.uv_layers.new(name=name)
    obj.data.uv_layers.active = obj.data.uv_layers.get(name)


def apply_top_uv_project(obj, frame_size_m, uv_name="UVMap"):
    ensure_uv_layer(obj, uv_name)
    mod=obj.modifiers.get("UVProject") or obj.modifiers.new("UVProject", type='UV_PROJECT')
    cam=bpy.data.objects.get("UVProjectorTop")
    if cam is None:
        cdat=bpy.data.cameras.new("UVProjectorTop"); cam=bpy.data.objects.new("UVProjectorTop", cdat)
        bpy.context.scene.collection.objects.link(cam)
    cam.data.type='ORTHO'; cam.data.ortho_scale=frame_size_m
    cam.location=(0,0,1.0); cam.rotation_euler=(0,0,0)
    if len(mod.projectors)<1: mod.projectors_add()
    mod.projectors[0].object=cam
    mod.aspect_x=1.0; mod.aspect_y=1.0; mod.uv_layer=uv_name


def make_white_material(name, rgba=(1,1,1,1)):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name); m.use_nodes=True
    nt=m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out=nt.nodes.new("ShaderNodeOutputMaterial"); out.location=(400,0)
    bs =nt.nodes.new("ShaderNodeBsdfPrincipled"); bs.location=(150,0)
    bs.inputs["Base Color"].default_value=rgba; bs.inputs["Roughness"].default_value=0.8
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])
    return m


def make_weft_band_material_from_path(img_path, band_index, total_bands=10):
    m = (bpy.data.materials.get(f"Mat_WeftBand_{band_index:02d}")
         or bpy.data.materials.new(f"Mat_WeftBand_{band_index:02d}"))
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (1150, 0)
    bs  = nt.nodes.new("ShaderNodeBsdfPrincipled"); bs.location = (900, 0)

    mix = nt.nodes.new("ShaderNodeMixRGB"); mix.location = (660, 0)
    mix.inputs["Color1"].default_value = WEFT_SIDE_COLOR

    tex = nt.nodes.new("ShaderNodeTexImage"); tex.location = (420, 120)
    img = bpy.data.images.load(str(img_path), check_existing=True)
    tex.image = img
    tex.extension = 'CLIP'
    try: tex.image.colorspace_settings.name = "sRGB"
    except: pass
    print(f"    [IMG] loaded: {img.filepath} size={getattr(img,'size',(-1,-1))}")

    mapn = nt.nodes.new("ShaderNodeMapping"); mapn.location = (200, 120)
    mapn.inputs["Scale"].default_value[0] = 1.0
    mapn.inputs["Scale"].default_value[1] = float(total_bands)
    mapn.inputs["Location"].default_value[0] = 0.0
    mapn.inputs["Location"].default_value[1] = float(-band_index)

    tco = nt.nodes.new("ShaderNodeTexCoord"); tco.location = (-20, 120)

    if DEBUG_FORCE_IMAGE_ON_TOP:
        mix.inputs["Fac"].default_value = 1.0
        print("    [DBG] DEBUG_FORCE_IMAGE_ON_TOP=True → top-mask bypass")
    else:
        geom = nt.nodes.new("ShaderNodeNewGeometry"); geom.location = (200, -220)
        dot  = nt.nodes.new("ShaderNodeVectorMath"); dot.location = (420, -220); dot.operation = 'DOT_PRODUCT'
        dot.inputs[1].default_value = (0.0, 0.0, 1.0)
        mr = nt.nodes.new("ShaderNodeMapRange"); mr.location = (640, -220)
        mr.inputs['From Min'].default_value = -1.0
        mr.inputs['From Max'].default_value =  1.0
        mr.inputs['To Min'].default_value   =  0.0
        mr.inputs['To Max'].default_value   =  1.0
        mr.clamp = True
        ramp = nt.nodes.new("ShaderNodeValToRGB"); ramp.location = (840, -220)
        ramp.color_ramp.elements[0].position = float(TOP_MASK_LOW)
        ramp.color_ramp.elements[1].position = float(TOP_MASK_HIGH)
        nt.links.new(geom.outputs["Normal"], dot.inputs[0])
        nt.links.new(dot.outputs["Value"],   mr.inputs['Value'])
        nt.links.new(mr.outputs["Result"],   ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"],  mix.inputs["Fac"])

    nt.links.new(tco.outputs["UV"],      mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])  # UVProject → Mapping
    nt.links.new(tex.outputs["Color"],   mix.inputs["Color2"])  # fac=1 → image

    nt.links.new(mix.outputs["Color"],   bs.inputs["Base Color"])
    nt.links.new(bs.outputs["BSDF"],     out.inputs["Surface"])
    return m

# =========================
# FILE DISCOVERY
# =========================

def load_h_bands(band_dir: Path, expect=10):
    folder = Path(band_dir)
    if not folder.exists():
        raise FileNotFoundError(f"폴더가 없습니다: {folder}")
    pat = re.compile(r"_H_(\d+)\.png$", re.IGNORECASE)
    found = []
    for p in folder.glob("*.png"):
        m = pat.search(p.name)
        if m: found.append((int(m.group(1)), p))
    if len(found) < expect:
        raise RuntimeError(f"H 밴드가 {len(found)}장만 있습니다. 10장 필요. ({folder})")
    found.sort(key=lambda x: x[0])
    paths = [p for _, p in found[:expect]]
    print(f"  [LOAD] {folder.name}: {len(paths)} H-bands → {[p.name for p in paths]}")
    return paths


def find_band_dirs(input_path: Path):
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"입력 경로가 없습니다: {p}")

    def has_h_series(dirpath: Path):
        return len(list(dirpath.glob("*_H_*.png"))) >= 10

    if p.is_dir() and has_h_series(p):
        return [p]

    band_dirs = []
    if p.is_dir():
        for child in p.iterdir():
            if child.is_dir() and has_h_series(child):
                band_dirs.append(child)
    band_dirs.sort(key=lambda d: d.name)
    if not band_dirs:
        raise RuntimeError(f"스와치 폴더를 찾지 못했습니다.(_H_*.png 10장 이상) in {p}")
    return band_dirs


def base_name_from_folder(folder_name: str) -> str:
    """Extract logical base (e.g., 'check_1' from 'check_1_rotate_crop_rot3')."""
    m = re.match(r"^(.*?)(_rotate_crop(?:_rot)?\d+)$", folder_name)
    if m:
        return m.group(1)
    m2 = re.match(r"^(.*?)(?:_[A-Za-z]+\d+)$", folder_name)
    if m2:
        return m2.group(1)
    return folder_name


def group_by_base(dirs):
    groups = {}
    for d in dirs:
        base = base_name_from_folder(d.name)
        groups.setdefault(base, []).append(d)
    for b in groups:
        groups[b].sort(key=lambda p: p.name)
    return groups

VARIANT_ORDER = [f"rot{k}" for k in range(1,6)] + [f"crop{k}" for k in range(1,6)]

def variant_score(path: Path) -> tuple:
    name = path.name
    for idx, tag in enumerate(VARIANT_ORDER):
        if name.endswith(tag):
            return (0, idx)  # 높은 우선순위
    return (1, name)        # 그 외는 뒤로


def pick_variants(dirs_for_base):
    mode = COMBO_VARIANT_PICK_MODE
    dirs_sorted = sorted(dirs_for_base, key=variant_score)
    if mode == 'best':
        return [dirs_sorted[0]]
    if mode == 'all':
        return dirs_sorted
    # firstN (default)
    return dirs_sorted[:max(1, int(VARIANTS_PER_BASE))]

# =========================
# BUILD & RENDER
# =========================

def setup_output_png(out_dir: Path, out_basename: str):
    from datetime import datetime
    scn = bpy.context.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (OUTPUT_BASENAME + "_") if OUTPUT_BASENAME else ""
    out_path = out_dir / f"{prefix}{out_basename}_{ts}.png"

    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode  = 'RGBA' if OUTPUT_ALPHA else 'RGB'
    scn.render.image_settings.color_depth = PNG_BIT_DEPTH
    scn.render.image_settings.compression = PNG_COMPRESSION
    if OUTPUT_ALPHA:
        try: scn.render.film_transparent = True
        except: pass
    scn.render.use_file_extension = True
    scn.render.filepath = str(out_path)


def build_scene_common():
    """Create geometry, UV, camera, light; return (warp_objs, weft_objs, frame_m, total_w, total_h)."""
    # init
    nuke_scene(keep_world=True); set_units_cm(); setup_renderer()

    sp_warp, sp_weft, crimp = pitch_m, pitch_m, crimp_amp
    if AUTO_CLEARANCE:
        min_warp_pitch=float(warp_w)*(1.0+CLEAR_MARGIN)
        min_weft_pitch=float(weft_w)*(1.0+CLEAR_MARGIN)
        if sp_warp<min_warp_pitch: sp_warp=min_warp_pitch
        if sp_weft<min_weft_pitch: sp_weft=min_weft_pitch
        min_center_gap=(float(warp_t)+float(weft_t))*(1.0+CLEAR_MARGIN)
        if 2.0*crimp<min_center_gap: crimp=0.5*min_center_gap

    total_w=warp_cnt*sp_warp; total_h=weft_cnt*sp_weft
    scn=bpy.context.scene
    col=bpy.data.collections.new("PlainWeave_Swatch"); scn.collection.children.link(col)

    # profiles
    prof_warp=make_rect_profile("Profile_Warp", warp_w, warp_t)
    prof_weft=make_rect_profile("Profile_Weft", weft_w, weft_t)
    try:
        prof_warp.hide_set(True); prof_weft.hide_set(True)
        prof_warp.hide_render=True; prof_weft.hide_render=True
    except: pass

    # curves
    mat_warp_white = make_white_material("Mat_Warp_White", WARP_WHITE)
    mat_dummy      = make_white_material("Mat_Dummy", (0.8,0.8,0.8,1.0))

    x0=-0.5*total_w+sp_warp*0.5; y0=-0.5*total_h+sp_weft*0.5
    warp_objs=[]; weft_objs=[]
    for i in range(warp_cnt):
        x=x0 + i*sp_warp
        o=build_axis_curve('warp', x, weft_cnt, sp_weft, i, mat_warp_white,
                           warp_w, warp_t, STEPS_PER_CELL, crimp, prof_warp)
        col.objects.link(o); warp_objs.append(o)
    for j in range(weft_cnt):
        y=y0 + j*sp_weft
        o=build_axis_curve('weft', y, warp_cnt, sp_warp, j, mat_dummy,
                           weft_w, weft_t, STEPS_PER_CELL, crimp, prof_weft)
        col.objects.link(o); weft_objs.append(o)

    # to mesh + UV projector
    warp_objs = convert_to_mesh(warp_objs)
    weft_objs = convert_to_mesh(weft_objs)
    frame_m = SWATCH_SIZE_CM*CM2M
    for o in warp_objs + weft_objs:
        apply_top_uv_project(o, frame_size_m=frame_m, uv_name="UVMap")

    # camera
    cam=bpy.data.objects.get("WeaveCam")
    if cam is None:
        cdat=bpy.data.cameras.new("WeaveCam"); cam=bpy.data.objects.new("WeaveCam", cdat)
        scn.collection.objects.link(cam)
    cam.data.type='ORTHO'; cam.data.ortho_scale=max(total_w,total_h)*1.08
    cam.location=(0,0,max(total_w,total_h)*2.0); cam.rotation_euler=(0,0,0)
    scn.camera=cam

    # world + light
    setup_world_and_single_light(total_w, total_h)

    print(f"[GEOM] pitch={pitch_m:.4f}m weft_w={weft_w:.4f}m warp_w={warp_w:.4f}m thick={weft_t:.4f}m")
    print(f"[GEOM] total_w={total_w:.4f}m total_h={total_h:.4f}m warp_cnt={warp_cnt} weft_cnt={weft_cnt}")

    return warp_objs, weft_objs, frame_m, total_w, total_h


def apply_weft_materials_from_paths(weft_objs, h_paths_rows):
    """Assign per-row materials from explicit h_paths_rows (len==10). Logs mapping."""
    assert len(h_paths_rows) == weft_cnt, f"expected {weft_cnt} rows"
    for j, o in enumerate(weft_objs):
        band_row = (weft_cnt-1 - j) if H_FILES_ARE_TOP_TO_BOTTOM else j
        img_path = h_paths_rows[band_row]
        print(f"    [H-MAP] row={j:02d} (band_row={band_row:02d}) -> {img_path}")
        mat = make_weft_band_material_from_path(img_path, band_index=j, total_bands=weft_cnt)
        o.data.materials[0] = mat


def build_and_render_from_hpaths(h_paths_rows, out_dir: Path, out_basename: str):
    warp_objs, weft_objs, frame_m, total_w, total_h = build_scene_common()
    apply_weft_materials_from_paths(weft_objs, h_paths_rows)

    if bpy.context.screen:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_perspective = 'CAMERA'
                        break

    print(f"[INFO] Swatch ready | Env={WORLD_STRENGTH}, Light={LIGHT_TOP_ENERGY_W}W, H={LIGHT_TOP_HEIGHT_M}m, weft_z_offset={WEFT_Z_OFFSET_M} m")
    setup_output_png(out_dir, out_basename)
    bpy.ops.render.render(write_still=True)


# =========================
# COMBO LOGIC
# =========================

def make_hpaths_for_single(dirA: Path):
    return load_h_bands(dirA, expect=weft_cnt)


def make_hpaths_for_combo_k(dirs: list[Path]):
    """Return list of 10 image paths alternating across K folders in cyclic order.
    Rule (1-based rows): rows 1..K use H_00 of each folder in order; rows K+1..2K use H_01; etc.
    """
    K = len(dirs)
    assert K in (2,3), "combo_k supports K=2 or K=3"
    H = {}
    for d in dirs:
        H[d] = load_h_bands(d, expect=weft_cnt)
    rows = []
    for j in range(weft_cnt):
        which = dirs[j % K]
        h_idx = j // K
        rows.append(H[which][h_idx])
    print(f"  [COMBO{K}] order: {[d.name for d in dirs]} → rows use indices per rule (j%{K}, j//{K}).")
    return rows


# =========================
# RUNNERS
# =========================

def run_single_by_folder(all_dirs):
    outdir = Path(OUTPUT_DIR) / "single_by_folder"
    print(f"\n=== SINGLE BY FOLDER === ({len(all_dirs)} dirs)")
    for k, d in enumerate(all_dirs, 1):
        print(f"[{k}/{len(all_dirs)}] {d}")
        hpaths = make_hpaths_for_single(d)
        safe = d.name
        build_and_render_from_hpaths(hpaths, outdir, safe)


def run_combo2_auto(groups):
    outdir = Path(OUTPUT_DIR) / "combo2"
    bases = sorted(groups.keys())
    pairs = list(combinations(bases, 2))
    print(f"\n=== COMBO2 AUTO === ({len(pairs)} base-pairs from bases: {bases})")
    for a, b in pairs:
        A = pick_variants(groups[a])
        B = pick_variants(groups[b])
        print(f"  pair: {a} × {b} → variantsA={ [p.name for p in A] } variantsB={ [p.name for p in B] }")
        combos = list(product(A, B))[:MAX_COMBO2_PER_PAIR]
        for da, db in combos:
            hpaths = make_hpaths_for_combo_k([da, db])
            outname = f"{a}__{da.name}__ALT2__{b}__{db.name}"
            build_and_render_from_hpaths(hpaths, outdir, outname)

    for tup in EXPLICIT_COMBO2:
        pa, pb = [Path(x) for x in tup]
        print(f"  pair (explicit): {pa.name} × {pb.name}")
        hpaths = make_hpaths_for_combo_k([pa, pb])
        outname = f"explicit__{pa.name}__ALT2__{pb.name}"
        build_and_render_from_hpaths(hpaths, outdir, outname)


def run_combo3_auto(groups):
    outdir = Path(OUTPUT_DIR) / "combo3"
    bases = sorted(groups.keys())
    triples = list(combinations(bases, 3))
    print(f"\n=== COMBO3 AUTO === ({len(triples)} base-triples from bases: {bases})")
    for a, b, c in triples:
        A = pick_variants(groups[a])
        B = pick_variants(groups[b])
        C = pick_variants(groups[c])
        print(f"  triple: {a} × {b} × {c} → A={ [p.name for p in A] } B={ [p.name for p in B] } C={ [p.name for p in C] }")
        combos = list(product(A, B, C))
        if len(combos) > MAX_COMBO3_PER_TRIPLE:
            combos = combos[:MAX_COMBO3_PER_TRIPLE]
        for da, db, dc in combos:
            hpaths = make_hpaths_for_combo_k([da, db, dc])
            outname = f"{a}__{da.name}__ALT3__{b}__{db.name}__{c}__{dc.name}"
            build_and_render_from_hpaths(hpaths, outdir, outname)

    for tup in EXPLICIT_COMBO3:
        pa, pb, pc = [Path(x) for x in tup]
        print(f"  triple (explicit): {pa.name} × {pb.name} × {pc.name}")
        hpaths = make_hpaths_for_combo_k([pa, pb, pc])
        outname = f"explicit__{pa.name}__ALT3__{pb.name}__{pc.name}"
        build_and_render_from_hpaths(hpaths, outdir, outname)


# =========================
# MAIN
# =========================

def main():
    all_dirs = find_band_dirs(Path(BAND_INPUT_PATH))
    print(f"[RUN] 총 {len(all_dirs)}개 스와치 폴더 발견 under {BAND_INPUT_PATH}")
    for d in all_dirs: print(f"   - {d}")

    groups = group_by_base(all_dirs)
    print("\n[BASE GROUPS]")
    for b, ds in groups.items():
        print(f"  {b}: {[p.name for p in ds]}")

    if RENDER_SINGLE_BY_FOLDER:
        run_single_by_folder(all_dirs)
    if RENDER_COMBO2_AUTO:
        run_combo2_auto(groups)
    if RENDER_COMBO3_AUTO:
        run_combo3_auto(groups)


if __name__ == "__main__":
    main()
