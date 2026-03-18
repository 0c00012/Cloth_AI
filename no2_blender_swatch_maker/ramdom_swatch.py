# -*- coding: utf-8 -*-
import bpy
import math
import os
import csv
import random
from math import sin, pi
from mathutils import Matrix, Vector

# ================================================================
#                      사용자 설정 (필수 확인)
# ================================================================
DO_RENDER        = True
OUTPUT_DIR       = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no2_blender_swatch_maker\render_output\ramdom_batch_0922"
FABRIC_PATHS     = [
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\0819_JH_crop\KakaoTalk_20230321_093803765_17_crop4.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\0819_JH_crop\KakaoTalk_20230321_093636783_01_crop1.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\0819_JH_crop\KakaoTalk_20230321_093803765_11_crop3.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\0819_JH_crop\KakaoTalk_20230321_093803765_01_crop5.png",
]

VARIATION_COUNT  = 50
GLOBAL_SEED      = 42     # None이면 매 실행마다 다른 결과. 고정 재현 원하면 정수 지정.

# -------- 기본 직물 해상도/엔진 ----------
USE_CYCLES       = True
SAMPLES          = 256
USE_DENOISER     = True
CYCLES_DEVICE    = 'CUDA'

# -------- 카메라/라이트 ----------
ADD_CAMERA_LIGHT = True   # ← 조명 "건드리지 않기" 기본
LIGHT_ENERGY      = 250.0
LIGHT_HEIGHT      = 1.0
LIGHT_SIZE_X_MULT = 1.2
LIGHT_SIZE_Y_MULT = 1.2

OBLIQUE_VIEW       = True
CAMERA_PERSPECTIVE = True
CAMERA_ELEV_DEG    = 60.0
CAMERA_AZIMUTH_DEG = 0.0
CAMERA_RADIUS_MULT = 2.2
CAMERA_LENS_MM     = 50.0
WEAVECAM_LOCATION  = (0.0, -4.3352, 7.6820)
WEAVECAM_LOOK_AT_TARGET = True
WEAVECAM_ROTATION_EULER_DEG = None

# -------- UV/텍스처 ----------
UV_PROJECT_FROM_TOP = True
TEX_REPEAT_RANGE    = (0.8, 2.2)   # 무작위 타일 스케일 범위
ROTATE_WEFT_90_PROB = 0.5          # 위사 텍스처 90° 회전 확률

# -------- 베이스 치수(랜덤은 이 주변에서 결정) ----------
WARP_COUNT   = 64
WEFT_COUNT   = 64

BASE_WARP_WIDTH  = 0.055
BASE_WARP_THICK  = 0.020
BASE_WEFT_WIDTH  = 0.060
BASE_WEFT_THICK  = 0.020

BASE_SPACING_WARP = 0.090
BASE_SPACING_WEFT = 0.060
BASE_CRIMP_AMP    = 0.0032

AUTO_CLEARANCE    = True
CLEARANCE_MARGIN  = 0.05

STEPS_PER_CELL    = 6
BEVEL_RES         = 3
RESOLUTION_U      = 24

WARP_COLOR        = (0.08, 0.08, 0.08, 1.0)
WEFT_COLOR        = (0.85, 0.85, 0.85, 1.0)

OBJECT_OFFSET     = (0.0, 0.1, 0.0)

# ================================================================
#                          유틸/내부 함수
# ================================================================
def clean_collection(name: str):
    scene = bpy.context.scene
    root = scene.collection
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        root.children.link(col)
    else:
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        for child in list(col.children):
            col.children.unlink(child)
    return col

def _safe_set_input(node, socket_names, value):
    if isinstance(socket_names, str):
        socket_names = [socket_names]
    for nm in socket_names:
        sock = node.inputs.get(nm)
        if sock is not None:
            try: sock.default_value = value
            except Exception: pass
            return True
    return False

def make_material(name, rgba):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (150, 0)
    _safe_set_input(bsdf, "Base Color", rgba)
    _safe_set_input(bsdf, "Roughness", 0.8)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def poly_curve(name, points):
    cu = bpy.data.curves.new(name=name, type='CURVE')
    cu.dimensions = '3D'; cu.fill_mode = 'FULL'
    spline = cu.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for i, (x, y, z) in enumerate(points):
        spline.points[i].co = (x, y, z, 1.0)
    return bpy.data.objects.new(name, cu)

def make_rect_profile(name, width, thickness):
    cu = bpy.data.curves.new(name=name, type='CURVE')
    cu.dimensions = '2D'; cu.fill_mode = 'BOTH'
    sp = cu.splines.new('POLY'); sp.use_cyclic_u = True
    hw, ht = float(width)*0.5, float(thickness)*0.5
    pts = [(-hw,-ht,0),(hw,-ht,0),(hw,ht,0),(-hw,ht,0)]
    sp.points.add(len(pts)-1)
    for i,(x,y,z) in enumerate(pts): sp.points[i].co=(x,y,z,1.0)
    obj = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def delete_object_if_exists(name):
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.data.objects.remove(obj, do_unlink=True)

def build_axis_curve(axis, fixed_pos, count_other, spacing_along, idx_self, mat,
                     width, thickness, steps_per_cell=6, crimp_amp=0.0032,
                     bevel_profile_obj=None):
    half_len = (count_other*spacing_along)*0.5
    total_steps = count_other*steps_per_cell
    pts=[]
    for s in range(total_steps+1):
        t = s/total_steps*(count_other*spacing_along)
        j = min(int(t//spacing_along), count_other-1)
        u = (t - j*spacing_along)/spacing_along
        if axis=='warp':
            x=fixed_pos; y=-half_len+t
            over=((idx_self+j)%2)==0; sign=+1 if over else -1
            z=sign*crimp_amp*sin(pi*u)
        else:
            x=-half_len+t; y=fixed_pos
            over=((j+idx_self)%2)==0; sign=-1 if over else +1
            z=sign*crimp_amp*sin(pi*u)
        pts.append((x,y,z))
    name=f"{'Warp' if axis=='warp' else 'Weft'}_{idx_self:02d}"
    obj = poly_curve(name, pts)
    obj.data.bevel_mode='OBJECT'
    if bevel_profile_obj is not None: obj.data.bevel_object = bevel_profile_obj
    obj.data.bevel_factor_start=0.0; obj.data.bevel_factor_end=1.0
    obj.data.bevel_resolution=BEVEL_RES; obj.data.resolution_u=RESOLUTION_U
    try: obj.data.twist_mode='Z_UP'
    except Exception: pass
    if obj.data.materials: obj.data.materials[0]=mat
    else: obj.data.materials.append(mat)
    return obj

def auto_clearance_adjust_rect(warp_width, weft_width, spacing_warp, spacing_weft,
                               crimp_amp, warp_thick, weft_thick, margin):
    min_warp_pitch = float(warp_width)*(1.0+margin)
    min_weft_pitch = float(weft_width)*(1.0+margin)
    if spacing_warp<min_warp_pitch: spacing_warp=min_warp_pitch
    if spacing_weft<min_weft_pitch: spacing_weft=min_weft_pitch
    min_center_gap = (float(warp_thick)+float(weft_thick))*(1.0+margin)
    if 2.0*crimp_amp<min_center_gap: crimp_amp=0.5*min_center_gap
    return spacing_warp, spacing_weft, crimp_amp

def setup_cycles_engine(use_cycles=True, samples=256, enable_gpu=True, device_type='CUDA', use_denoiser=True):
    scene=bpy.context.scene
    if use_cycles:
        scene.render.engine='CYCLES'
        scene.cycles.samples=samples
        scene.render.film_transparent=True
        scene.view_settings.exposure=0.0
        if hasattr(scene.cycles,"use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling=True
            scene.cycles.adaptive_threshold=0.01
        if hasattr(scene.cycles,"use_light_tree"):
            scene.cycles.use_light_tree=True
        if use_denoiser:
            try:
                if hasattr(scene.cycles,"denoiser"):
                    scene.cycles.denoiser='OPTIX'
                bpy.context.view_layer.cycles.use_denoising=True
            except Exception:
                try:
                    if hasattr(scene.cycles,"denoiser"):
                        scene.cycles.denoiser='OPENIMAGEDENOISE'
                    bpy.context.view_layer.cycles.use_denoising=True
                except Exception: pass
        if enable_gpu:
            try:
                prefs=bpy.context.preferences
                cprefs=prefs.addons['cycles'].preferences
                cprefs.use_persistent_data=True
                cprefs.compute_device_type=device_type
                cprefs.get_devices()
                for dev in cprefs.devices: dev.use=True
                scene.cycles.device='GPU'
            except Exception as e:
                print(f"[WARN] Cycles GPU 설정 실패: {e}")
    else:
        scene.render.engine='BLENDER_EEVEE'

def make_image_material(name, image_path, repeat=(1.0,1.0), rotate_deg=0.0, roughness=1.0):
    img=None
    try: img=bpy.data.images.load(image_path, check_existing=True)
    except Exception as e: print(f"[WARN] 이미지 로드 실패: {image_path} ({e})")
    mat=bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes=True; nt=mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out=nt.nodes.new("ShaderNodeOutputMaterial"); out.location=(1000,0)
    bsdf=nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location=(760,0)
    tex=nt.nodes.new("ShaderNodeTexImage"); tex.location=(540,0)
    mapn=nt.nodes.new("ShaderNodeMapping"); mapn.location=(320,0)
    tco=nt.nodes.new("ShaderNodeTexCoord"); tco.location=(100,0)
    if img is not None:
        tex.image=img; tex.extension='REPEAT'
        try: tex.image.colorspace_settings.name="sRGB"
        except Exception: pass
    bsdf.inputs["Roughness"].default_value=roughness
    mapn.inputs["Scale"].default_value[0]=max(1e-6,float(repeat[0]))
    mapn.inputs["Scale"].default_value[1]=max(1e-6,float(repeat[1]))
    mapn.inputs["Rotation"].default_value[2]=math.radians(float(rotate_deg))
    nt.links.new(tco.outputs.get("UV") or tco.outputs["Generated"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def ensure_uv_layer(obj, name="UVMap"):
    if obj.type!='MESH': return
    names=[l.name for l in obj.data.uv_layers]
    if name not in names: obj.data.uv_layers.new(name=name)
    obj.data.uv_layers.active=obj.data.uv_layers.get(name)

def ensure_uv_project_modifier(obj, projector_obj, uv_name="UVMap"):
    if obj.type!='MESH': return
    ensure_uv_layer(obj, uv_name)
    mod=obj.modifiers.get("UVProject") or obj.modifiers.new("UVProject", type='UV_PROJECT')
    if len(mod.projectors)<1: mod.projectors_add()
    mod.projectors[0].object=projector_obj
    mod.uv_layer=uv_name
    mod.aspect_x=1.0; mod.aspect_y=1.0

def convert_to_mesh_if_needed(objs):
    need = any(o and o.type!='MESH' for o in objs)
    if not need: return
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        if o: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.convert(target='MESH')

def setup_camera_and_top_light(add_camera_light=True, total_w=1.0, total_h=1.0, target=(0,0,0)):
    scene=bpy.context.scene
    smax=max(float(total_w), float(total_h))

    cam=bpy.data.objects.get("WeaveCam")
    if cam is None:
        cam_data=bpy.data.cameras.new("WeaveCam")
        cam=bpy.data.objects.new("WeaveCam", cam_data)
        scene.collection.objects.link(cam)
    cam.rotation_mode='XYZ'
    if hasattr(cam.data,"shift_x"): cam.data.shift_x=0.0
    if hasattr(cam.data,"shift_y"): cam.data.shift_y=0.0
    cam.data.clip_start=0.001; cam.data.clip_end=100.0

    cam.data.type='PERSP' if (OBLIQUE_VIEW and CAMERA_PERSPECTIVE) else 'ORTHO'
    if cam.data.type=='ORTHO': cam.data.ortho_scale=smax*1.15
    if hasattr(cam.data,"lens"): cam.data.lens=float(CAMERA_LENS_MM)

    if WEAVECAM_LOCATION is not None:
        cam.location=(float(WEAVECAM_LOCATION[0]), float(WEAVECAM_LOCATION[1]), float(WEAVECAM_LOCATION[2]))
    else:
        elev=math.radians(float(CAMERA_ELEV_DEG))
        az  =math.radians(float(CAMERA_AZIMUTH_DEG))
        r   =max(0.25,float(CAMERA_RADIUS_MULT)*smax)*0.7
        base_x=target[0]+r*math.sin(az)*math.cos(elev)
        base_y=target[1]-r*math.cos(az)*math.cos(elev)
        base_z=target[2]+r*math.sin(elev)
        cam.location=(base_x,base_y,base_z)

    if WEAVECAM_ROTATION_EULER_DEG is not None:
        rx,ry,rz=[math.radians(float(a)) for a in WEAVECAM_ROTATION_EULER_DEG]
        cam.rotation_euler=(rx,ry,rz)
    elif WEAVECAM_LOOK_AT_TARGET:
        forward=(Vector(target)-Vector(cam.location)).normalized()
        world_up=Vector((0,0,1))
        right=forward.cross(world_up).normalized()
        if right.length<1e-6:
            world_up=Vector((0,1,0))
            right=forward.cross(world_up).normalized()
        up=right.cross(forward).normalized()
        rot_mat=Matrix(((right.x,up.x,-forward.x),
                        (right.y,up.y,-forward.y),
                        (right.z,up.z,-forward.z)))
        cam.rotation_euler=rot_mat.to_euler('XYZ')

    light=None
    if add_camera_light:
        keep="Light"
        for obj in list(scene.objects):
            if obj.type=='LIGHT' and obj.name!=keep:
                bpy.data.objects.remove(obj, do_unlink=True)
        light=bpy.data.objects.get(keep)
        if light is None:
            light_data=bpy.data.lights.new(name=keep, type='AREA')
            light=bpy.data.objects.new(name=keep, object_data=light_data)
            scene.collection.objects.link(light)
        light.data.type='AREA'; light.data.shape='RECTANGLE'
        light.data.size=max(smax*LIGHT_SIZE_X_MULT,0.05)
        light.data.size_y=max(smax*LIGHT_SIZE_Y_MULT,0.05)
        light.rotation_mode='XYZ'; light.rotation_euler=(0,0,0)
        light.location=(float(target[0]), float(target[1]), float(LIGHT_HEIGHT))
        light.data.energy=max(float(LIGHT_ENERGY),0.0)
        try: light.data.use_contact_shadow=True
        except Exception: pass

    scene.camera=cam
    return cam, light

def get_or_create_uv_projector(total_w, total_h, target=(0,0,0), name="UVProjectorTop"):
    smax=max(float(total_w), float(total_h))
    cam=bpy.data.objects.get(name)
    if cam is None:
        cam_data=bpy.data.cameras.new(name)
        cam=bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    cam.data.type='ORTHO'
    cam.data.ortho_scale=smax*1.5
    cam.location=(float(target[0]), float(target[1]), float(target[2])+1.0)
    cam.rotation_mode='XYZ'; cam.rotation_euler=(0,0,0)
    cam.data.clip_start=0.001; cam.data.clip_end=100.0
    return cam

def generate_plain_weave(
    warp_count, weft_count,
    warp_width, weft_width,
    warp_thick, weft_thick,
    spacing_warp, spacing_weft,
    crimp_amp,
    auto_clearance=True, clearance_margin=0.05,
    steps_per_cell=6, bevel_res=3, resolution_u=24,
    warp_color=(0.08, 0.08, 0.08, 1.0),
    weft_color=(0.85, 0.85, 0.85, 1.0)
):
    global BEVEL_RES, RESOLUTION_U, STEPS_PER_CELL
    BEVEL_RES=bevel_res; RESOLUTION_U=resolution_u; STEPS_PER_CELL=steps_per_cell
    if auto_clearance:
        spacing_warp, spacing_weft, crimp_amp = auto_clearance_adjust_rect(
            warp_width, weft_width, spacing_warp, spacing_weft,
            crimp_amp, warp_thick, weft_thick, clearance_margin
        )
    col=clean_collection("PlainWeave_Swatch")

    # 프로파일(폭/두께 갱신을 위해 항상 재생성)
    delete_object_if_exists("Profile_Warp_Rect")
    delete_object_if_exists("Profile_Weft_Rect")
    prof_warp=make_rect_profile("Profile_Warp_Rect", warp_width, warp_thick)
    prof_weft=make_rect_profile("Profile_Weft_Rect", weft_width, weft_thick)
    try:
        prof_warp.hide_set(True); prof_weft.hide_set(True)
        prof_warp.hide_render=True; prof_weft.hide_render=True
    except Exception: pass

    mat_warp=make_material("Mat_Warp_Base", WARP_COLOR)
    mat_weft=make_material("Mat_Weft_Base", WEFT_COLOR)

    total_w=warp_count*spacing_warp
    total_h=weft_count*spacing_weft
    x0=-0.5*total_w + spacing_warp*0.5
    y0=-0.5*total_h + spacing_weft*0.5

    warp_objs=[]; weft_objs=[]
    for i in range(warp_count):
        x=x0 + i*spacing_warp
        obj=build_axis_curve('warp', x, weft_count, spacing_weft, i, mat_warp,
                             warp_width, warp_thick, steps_per_cell, crimp_amp, prof_warp)
        col.objects.link(obj); warp_objs.append(obj)
    for j in range(weft_count):
        y=y0 + j*spacing_weft
        obj=build_axis_curve('weft', y, warp_count, spacing_warp, j, mat_weft,
                             weft_width, weft_thick, steps_per_cell, crimp_amp, prof_weft)
        col.objects.link(obj); weft_objs.append(obj)

    return {"collection": col, "total_w": total_w, "total_h": total_h,
            "warp_objs": warp_objs, "weft_objs": weft_objs}

def build_material_bank(paths, repeat_w=(1.0,1.0), repeat_we=(1.0,1.0), rotate_we_deg=0.0,
                        prefix_w="Mat_Warp_IMG_", prefix_we="Mat_Weft_IMG_"):
    warp_mats=[]; weft_mats=[]
    for idx, p in enumerate(paths):
        if p is None: continue
        mw = make_image_material(f"{prefix_w}{idx}", p, repeat=repeat_w, rotate_deg=0.0)
        me = make_image_material(f"{prefix_we}{idx}", p, repeat=repeat_we, rotate_deg=rotate_we_deg)
        warp_mats.append(mw); weft_mats.append(me)
    return warp_mats, weft_mats

def ensure_uv_project_for_objects(objs, projector):
    convert_to_mesh_if_needed(objs)
    for o in objs:
        if o and o.type=='MESH':
            ensure_uv_project_modifier(o, projector, uv_name="UVMap")

def apply_materials_with_index_maps(warp_objs, weft_objs, warp_mats, weft_mats, warp_map, weft_map):
    # warp_map, weft_map: 각 스트랜드별 재질 index 리스트
    for i, o in enumerate(warp_objs):
        if not o or o.type!='MESH': continue
        idx = warp_map[i] % max(1, len(warp_mats))
        mat = warp_mats[idx]
        if o.data.materials: o.data.materials[0]=mat
        else: o.data.materials.append(mat)
    for j, o in enumerate(weft_objs):
        if not o or o.type!='MESH': continue
        idx = weft_map[j] % max(1, len(weft_mats))
        mat = weft_mats[idx]
        if o.data.materials: o.data.materials[0]=mat
        else: o.data.materials.append(mat)

# ------------------- 랜덤 패턴 생성 -------------------
def make_index_sequence(pattern, need_n, count, rng):
    """
    pattern: "constant" | "alt" | "stripe" | "random" | "blocks"
    """
    seq = [0]*count
    if need_n <= 0:
        return seq
    if pattern == "constant":
        k = rng.randrange(need_n)
        seq = [k]*count
    elif pattern == "alt":
        offset = rng.randrange(need_n)
        for i in range(count):
            seq[i] = (i + offset) % need_n
    elif pattern == "stripe":
        width = rng.randint(1, 8)  # 줄무늬 폭
        offset = rng.randrange(need_n)
        for i in range(count):
            seq[i] = ((i // width) + offset) % need_n
    elif pattern == "random":
        for i in range(count):
            seq[i] = rng.randrange(need_n)
    elif pattern == "blocks":
        # need_n개 순열을 블록 단위로 반복
        perm = list(range(need_n))
        rng.shuffle(perm)
        block = []
        # 블록 폭 무작위
        block_w = rng.randint(2, max(2, need_n * 2))
        while len(block) < count:
            for p in perm:
                block.extend([p]*block_w)
                if len(block) >= count: break
        seq = block[:count]
    else:
        # fallback
        for i in range(count):
            seq[i] = i % need_n
    return seq

# ================================================================
#                             메인
# ================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 메시만 제거(조명/카메라/기타는 유지)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH'); bpy.ops.object.delete()

    setup_cycles_engine(USE_CYCLES, SAMPLES, enable_gpu=True, device_type=CYCLES_DEVICE, use_denoiser=USE_DENOISER)

    # 3D뷰 카메라 시점
    try:
        scr=bpy.context.screen
        if scr:
            for area in scr.areas:
                if area.type=='VIEW_3D':
                    for space in area.spaces:
                        if space.type=='VIEW_3D':
                            space.region_3d.view_perspective='CAMERA'; break
    except Exception:
        pass

    # 로그 파일 준비
    csv_path = os.path.join(OUTPUT_DIR, "variations_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx","need_n","fabrics_used_indices","warp_width","warp_thick","weft_width","weft_thick",
            "spacing_warp","spacing_weft","crimp_amp","repeat_w_u","repeat_w_v","repeat_we_u","repeat_we_v",
            "rotate_weft_90","pattern_warp","pattern_weft"
        ])

    # 전역 RNG
    base_rng = random.Random(GLOBAL_SEED) if GLOBAL_SEED is not None else random.Random()

    for vi in range(VARIATION_COUNT):
        rng = random.Random(base_rng.randrange(1<<30))  # 각 변형별 독립 난수

        # ---- 1) 사용할 원단 개수/순서 ----
        max_n = min(len(FABRIC_PATHS), 4)
        need_n = rng.randint(1, max_n)
        chosen_indices = list(range(len(FABRIC_PATHS)))
        rng.shuffle(chosen_indices)
        chosen_indices = chosen_indices[:need_n]
        chosen_paths = [FABRIC_PATHS[i] for i in chosen_indices]

        # ---- 2) 치수 랜덤(폭/두께/간격/크림프) ----
        warp_width = BASE_WARP_WIDTH  * rng.uniform(0.75, 1.35)
        weft_width = BASE_WEFT_WIDTH  * rng.uniform(0.75, 1.35)
        warp_thick = BASE_WARP_THICK  * rng.uniform(0.70, 1.30)
        weft_thick = BASE_WEFT_THICK  * rng.uniform(0.70, 1.30)

        spacing_warp = BASE_SPACING_WARP * rng.uniform(0.90, 1.15)
        spacing_weft = BASE_SPACING_WEFT * rng.uniform(0.90, 1.15)
        crimp_amp    = BASE_CRIMP_AMP   * rng.uniform(0.75, 1.50)

        # ---- 3) 직물 생성 (이번 변형에 맞춰 새로 생성) ----
        result = generate_plain_weave(
            warp_count=WARP_COUNT, weft_count=WEFT_COUNT,
            warp_width=warp_width, weft_width=weft_width,
            warp_thick=warp_thick, weft_thick=weft_thick,
            spacing_warp=spacing_warp, spacing_weft=spacing_weft,
            crimp_amp=crimp_amp,
            auto_clearance=AUTO_CLEARANCE, clearance_margin=CLEARANCE_MARGIN,
            steps_per_cell=STEPS_PER_CELL, bevel_res=BEVEL_RES, resolution_u=RESOLUTION_U,
            warp_color=WARP_COLOR, weft_color=WEFT_COLOR
        )

        # 위치 오프셋 적용
        all_objs = result["warp_objs"] + result["weft_objs"]
        for o in all_objs:
            if o:
                o.location = (o.location.x + OBJECT_OFFSET[0],
                              o.location.y + OBJECT_OFFSET[1],
                              o.location.z + OBJECT_OFFSET[2])
        target = OBJECT_OFFSET

        # 카메라/라이트(라이트는 기본 False로)
        cam, _ = setup_camera_and_top_light(ADD_CAMERA_LIGHT, result["total_w"], result["total_h"], target)
        bpy.context.scene.camera = cam

        # UV 프로젝터
        projector = get_or_create_uv_projector(result["total_w"], result["total_h"], target) if UV_PROJECT_FROM_TOP else cam

        # ---- 4) 텍스처 옵션 랜덤 ----
        repeat_w = (rng.uniform(*TEX_REPEAT_RANGE), rng.uniform(*TEX_REPEAT_RANGE))
        repeat_we = (rng.uniform(*TEX_REPEAT_RANGE), rng.uniform(*TEX_REPEAT_RANGE))
        rotate_we_deg = 90.0 if (rng.random() < ROTATE_WEFT_90_PROB) else 0.0

        # 재질 뱅크(이름은 고정 → 매 변형마다 노드가 갱신되어 중복 생성 방지)
        warp_mats_all, weft_mats_all = build_material_bank(
            chosen_paths, repeat_w=repeat_w, repeat_we=repeat_we, rotate_we_deg=rotate_we_deg
        )

        # ---- 5) 패턴 타입 랜덤 선택 & 인덱스 시퀀스 생성 ----
        pat_types = ["constant", "alt", "stripe", "random", "blocks"]
        pat_warp = rng.choice(pat_types)
        pat_weft = rng.choice(pat_types)

        warp_seq = make_index_sequence(pat_warp, need_n, WARP_COUNT, rng)
        weft_seq = make_index_sequence(pat_weft, need_n, WEFT_COUNT, rng)

        # 메시 변환 & UV Project 적용
        ensure_uv_project_for_objects(result["warp_objs"] + result["weft_objs"], projector)
        # 재질 할당
        apply_materials_with_index_maps(result["warp_objs"], result["weft_objs"],
                                        warp_mats_all, weft_mats_all, warp_seq, weft_seq)

        # ---- 6) 렌더 & 로그 ----
        fname = f"swatch_v{vi:02d}_n{need_n}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)
        if DO_RENDER:
            bpy.context.scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            print(f"[INFO] Render saved: {out_path}")
        else:
            print(f"[INFO] Prepared (미렌더): {out_path}")

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                vi, need_n, chosen_indices,
                round(warp_width,6), round(warp_thick,6), round(weft_width,6), round(weft_thick,6),
                round(spacing_warp,6), round(spacing_weft,6), round(crimp_amp,6),
                round(repeat_w[0],4), round(repeat_w[1],4), round(repeat_we[0],4), round(repeat_we[1],4),
                int(rotate_we_deg==90.0), pat_warp, pat_weft
            ])

    print(f"[INFO] Done. CSV log: {csv_path}")

# ================================================================
if __name__ == "__main__":
    main()
