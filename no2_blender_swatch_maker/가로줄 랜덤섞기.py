# -*- coding: utf-8 -*-
# slices_10h_10v 아래 여러 스와치 폴더에서 서로 다른 두 종류(A/B)를 자동으로 선택해
# 경사=흰색 고정, 위사만 줄 단위로 A/B를 섞어 다양한 전략으로 렌더/PNG 저장.

import bpy, re, os, random, itertools
from pathlib import Path
from math import sin, pi

# ================== 입력 루트/페어 선택 ==================
# slices_10h_10v 루트: 이 아래에 스와치별 하위폴더들이 있고, 각 폴더에 *_H_00..09.png가 있어야 함
SLICES_ROOT_DIR = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\1110_2_crop_size_1010\slices_10h_10v"

# 페어 선택 모드
USE_ALL_PAIRS          = True     # True면 모든 하위폴더 조합(NC2)을 처리
RANDOM_SAMPLE_PAIRS_K  = None     # 예: 5 → 전체 조합 중 5페어만 무작위 선택 (USE_ALL_PAIRS가 False일 때 사용)
EXPLICIT_PAIR_NAMES    = []       # 예: [("all-over_pattern_rotate_crop_rot1","check_1_rotate_crop_rot1"), ...]
                                  # 이름은 하위폴더명과 정확히 일치

# ================== 출력/렌더 공통 설정 ==================
SAVE_PNG         = True
OUTPUT_DIR       = r"C:\Users\_idal\PycharmProjects\Cloth_AI\renders"
OUTPUT_BASENAME  = "mix"          # 파일명 접두어(선택)
OUTPUT_ALPHA     = True
PNG_BIT_DEPTH    = '8'            # '8' or '16'
PNG_COMPRESSION  = 15             # 0~100

SWATCH_SIZE_CM = 10.0
CELL_PITCH_CM  = 1.0
DO_RENDER_NOW  = True

WORLD_STRENGTH       = 0.15
WORLD_COLOR          = (1.0, 1.0, 1.0, 1.0)
LIGHT_SIZE_MULT      = 1.25
CM_EXPOSURE          = 0.05
LIGHT_TOP_ENERGY_W   = 22.0
LIGHT_TOP_HEIGHT_M   = 1.5

USE_CYCLES  = True
SAMPLES     = 128
RESOLUTION  = (1500, 1500)

# 실 형상/크림프
WIDTH_RATIO = 0.60
THICK_RATIO = 0.25
CRIMP_RATIO = 0.32
CRIMP_SCALE = 0.7
CRIMP_ABS_CM = None

AUTO_CLEARANCE = False
CLEAR_MARGIN   = 0.05
STEPS_PER_CELL = 6
BEVEL_RES      = 3
RESOLUTION_U   = 24
WEFT_Z_OFFSET_M = 0.0003

# 색
WARP_WHITE      = (1,1,1,1)    # 경사=항상 흰색
WEFT_SIDE_COLOR = (0,0,0,0)

# H_00이 최상단이면 True (위→아래 역매핑)
H_FILES_ARE_TOP_TO_BOTTOM = True

# ===== 믹스 전략 =====
NUM_VARIATIONS_PER_PAIR = 6      # 페어당 생성할 랜덤 조합 수
RANDOM_SEED             = 2025   # 재현성 원하면 고정, None면 매번 달라짐
STRATEGY_POOL = [
    "ALT_A",              # A,B,A,B,…
    "ALT_B",              # B,A,B,A,…
    "RANDOM_UNIFORM",     # 각 줄 독립 50:50
    "RANDOM_BALANCED",    # 5개씩 균형 맞춰 무작위 배치
    "BLOCKS",             # 길이 1~3의 블록 단위로 A/B 번갈아
    "TOP_A_BOTTOM_B",     # 상단 5줄 A, 하단 5줄 B
    "TOP_B_BOTTOM_A",     # 상단 5줄 B, 하단 5줄 A
    "A2B2"                # AA,BB,AA,BB,…
]

# ================== 파생 값 ==================
CM2M    = 0.01
pitch_m = CELL_PITCH_CM * CM2M
warp_cnt = weft_cnt = int(round(SWATCH_SIZE_CM / CELL_PITCH_CM))

warp_w   = pitch_m * WIDTH_RATIO
weft_w   = pitch_m * WIDTH_RATIO
warp_t   = pitch_m * THICK_RATIO
weft_t   = pitch_m * THICK_RATIO

def _compute_crimp_amp():
    if CRIMP_ABS_CM is not None:
        return float(CRIMP_ABS_CM) * CM2M
    return pitch_m * CRIMP_RATIO * CRIMP_SCALE
crimp_amp = _compute_crimp_amp()

# ================== PNG 출력 세팅 ==================
def setup_output_png(out_basename: str):
    from datetime import datetime
    scn = bpy.context.scene
    out_dir = Path(OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
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

# ================== 씬/렌더 세팅 ==================
def nuke_scene(keep_world=True):
    try:
        if bpy.ops.object.mode_set.poll(): bpy.ops.object.mode_set(mode='OBJECT')
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
        if n.bl_idname == "ShaderNodeBackground":
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

# ================== 밴드 로드/지오메트리/재료 ==================
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
    return [p for _, p in found[:expect]]

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
    tex.image = bpy.data.images.load(str(img_path), check_existing=True)
    tex.extension = 'CLIP'
    try: tex.image.colorspace_settings.name = "sRGB"
    except: pass

    mapn = nt.nodes.new("ShaderNodeMapping"); mapn.location = (200, 120)
    mapn.inputs["Scale"].default_value[0] = 1.0
    mapn.inputs["Scale"].default_value[1] = float(total_bands)
    mapn.inputs["Location"].default_value[0] = 0.0
    mapn.inputs["Location"].default_value[1] = float(-band_index)

    tco = nt.nodes.new("ShaderNodeTexCoord"); tco.location = (-20, 120)

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
    ramp.color_ramp.elements[0].position = 0.65
    ramp.color_ramp.elements[1].position = 0.95

    nt.links.new(tco.outputs["UV"],      mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"],   mix.inputs["Color2"])

    nt.links.new(geom.outputs["Normal"], dot.inputs[0])
    nt.links.new(dot.outputs["Value"],   mr.inputs["Value"])
    nt.links.new(mr.outputs["Result"],   ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"],  mix.inputs["Fac"])

    nt.links.new(mix.outputs["Color"],   bs.inputs["Base Color"])
    nt.links.new(bs.outputs["BSDF"],     out.inputs["Surface"])
    return m

# ================== 믹스 전략 로직 ==================
def make_mix_assignment(weft_count:int, strategy:str, rng:random.Random):
    """길이 weft_count의 리스트 반환. 각 원소는 'A' 또는 'B'."""
    if strategy == "ALT_A":
        return ['A' if i%2==0 else 'B' for i in range(weft_count)]
    if strategy == "ALT_B":
        return ['B' if i%2==0 else 'A' for i in range(weft_count)]
    if strategy == "RANDOM_UNIFORM":
        return [rng.choice(['A','B']) for _ in range(weft_count)]
    if strategy == "RANDOM_BALANCED":
        arr = ['A']*(weft_count//2) + ['B']*(weft_count - weft_count//2)
        rng.shuffle(arr); return arr
    if strategy == "BLOCKS":
        arr=[]; cur=rng.choice(['A','B'])
        while len(arr)<weft_count:
            block = rng.randint(1,3)
            arr.extend([cur]*block)
            cur = 'B' if cur=='A' else 'A'
        return arr[:weft_count]
    if strategy == "TOP_A_BOTTOM_B":
        return ['A']* (weft_count//2) + ['B']*(weft_count - weft_count//2)
    if strategy == "TOP_B_BOTTOM_A":
        return ['B']* (weft_count//2) + ['A']*(weft_count - weft_count//2)
    if strategy == "A2B2":
        arr=[]
        while len(arr)<weft_count:
            arr.extend(['A','A','B','B'])
        return arr[:weft_count]
    # fallback
    return ['A' if i%2==0 else 'B' for i in range(weft_count)]

# ================== 스와치 빌드/렌더 ==================
def build_and_render_mixed_swatch(h_paths_A, h_paths_B, assignment, tag_basename):
    nuke_scene(keep_world=True); set_units_cm(); setup_renderer()

    # 피치/사이즈
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

    # 프로파일
    prof_warp=make_rect_profile("Profile_Warp", warp_w, warp_t)
    prof_weft=make_rect_profile("Profile_Weft", weft_w, weft_t)
    try:
        prof_warp.hide_set(True); prof_weft.hide_set(True)
        prof_warp.hide_render=True; prof_weft.hide_render=True
    except: pass

    # 재료
    mat_warp_white = make_white_material("Mat_Warp_White", WARP_WHITE)
    mat_dummy      = make_white_material("Mat_Dummy", (0.8,0.8,0.8,1.0))

    # 경사/위사 곡선
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

    # 메쉬 변환 + UVProject
    warp_objs = convert_to_mesh(warp_objs)
    weft_objs = convert_to_mesh(weft_objs)
    frame_m = SWATCH_SIZE_CM*CM2M
    for o in warp_objs + weft_objs:
        apply_top_uv_project(o, frame_size_m=frame_m, uv_name="UVMap")

    # 위사 줄마다 A/B 이미지 선택
    for j, o in enumerate(weft_objs):
        row_idx = (weft_cnt-1 - j) if H_FILES_ARE_TOP_TO_BOTTOM else j
        img_path = h_paths_A[row_idx] if assignment[j]=='A' else h_paths_B[row_idx]
        mat = make_weft_band_material_from_path(img_path, band_index=j, total_bands=weft_cnt)
        o.data.materials[0] = mat

    # 카메라/라이트
    cam=bpy.data.objects.get("WeaveCam")
    if cam is None:
        cdat=bpy.data.cameras.new("WeaveCam"); cam=bpy.data.objects.new("WeaveCam", cdat)
        scn.collection.objects.link(cam)
    cam.data.type='ORTHO'; cam.data.ortho_scale=max(total_w,total_h)*1.08
    cam.location=(0,0,max(total_w,total_h)*2.0); cam.rotation_euler=(0,0,0)
    scn.camera=cam
    setup_world_and_single_light(total_w, total_h)

    # 렌더/저장
    if DO_RENDER_NOW:
        if SAVE_PNG:
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in tag_basename)
            setup_output_png(safe)
            bpy.ops.render.render(write_still=True)
        else:
            bpy.ops.render.render(write_still=False)

# ================== 스와치 폴더 탐색/페어 생성 ==================
def list_valid_swatch_dirs(root: Path):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"루트 폴더 없음: {root}")
    def is_valid_dir(p: Path):
        return p.is_dir() and len(list(p.glob("*_H_*.png"))) >= 10
    dirs = [p for p in root.iterdir() if is_valid_dir(p)]
    dirs.sort(key=lambda d: d.name)
    if len(dirs) < 2:
        raise RuntimeError(f"유효 스와치 폴더가 2개 미만입니다. (found={len(dirs)})")
    return dirs

def resolve_pairs(all_dirs, use_all=True, sample_k=None, explicit_names=None, rng=None):
    if explicit_names:
        name2path = {d.name: d for d in all_dirs}
        pairs = []
        for a,b in explicit_names:
            if a in name2path and b in name2path and a!=b:
                pairs.append((name2path[a], name2path[b]))
        if not pairs:
            raise RuntimeError("EXPLICIT_PAIR_NAMES에 유효한 쌍이 없습니다.")
        return pairs
    combos = list(itertools.combinations(all_dirs, 2))  # (A,B) unordered
    if use_all:
        return combos
    if sample_k is None or sample_k <= 0 or sample_k >= len(combos):
        return combos
    rng = rng or random.Random()
    rng.shuffle(combos)
    return combos[:sample_k]

# ================== 메인(모든 페어 순회) ==================
def main():
    rng = random.Random(RANDOM_SEED) if RANDOM_SEED is not None else random.Random()

    swatch_dirs = list_valid_swatch_dirs(SLICES_ROOT_DIR)
    pairs = resolve_pairs(
        swatch_dirs,
        use_all=USE_ALL_PAIRS,
        sample_k=RANDOM_SAMPLE_PAIRS_K,
        explicit_names=EXPLICIT_PAIR_NAMES,
        rng=rng
    )

    print(f"[RUN] 유효 스와치 폴더 {len(swatch_dirs)}개, 처리할 페어 {len(pairs)}개")

    for pi, (dirA, dirB) in enumerate(pairs, 1):
        hA = load_h_bands(dirA, expect=weft_cnt)
        hB = load_h_bands(dirB, expect=weft_cnt)
        print(f"\n[{pi}/{len(pairs)}] Pair: {dirA.name}  ×  {dirB.name}")

        for vi in range(1, NUM_VARIATIONS_PER_PAIR+1):
            strategy = rng.choice(STRATEGY_POOL)
            assign = make_mix_assignment(weft_cnt, strategy, rng)
            tag = f"pair{pi:02d}_{dirA.name}__{dirB.name}_mix{vi:02d}_{strategy}_" + "".join(assign)
            print(f"  -> {strategy}: {assign}")
            build_and_render_mixed_swatch(hA, hB, assign, tag)

    print("\n✅ 모든 페어 렌더 완료.")

# ===== 실행 =====
if __name__ == "__main__":
    main()
