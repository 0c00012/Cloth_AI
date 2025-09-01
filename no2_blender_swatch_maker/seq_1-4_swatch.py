import bpy
import math
import os
from math import sin, pi
from mathutils import Matrix, Vector

# ========================= (렌더/입출력) =========================
DO_RENDER   = True   # ▶ 순차 렌더 실행 여부
OUTPUT_DIR  = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no2_blender_swatch_maker\render_output\batch"

# ========================= (원단 이미지 경로) ====================
# 제공하신 4개 원단을 순서대로 사용합니다.
FABRIC_PATHS = [
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt\KakaoTalk_20230321_093636783_18_crop3.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt\KakaoTalk_20230321_093636783_14_crop3.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt\KakaoTalk_20230321_093636783_21_crop4.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt\KakaoTalk_20230321_093727482_14_crop5.png",
]

# ========================= (직물 매개변수) =======================
WARP_COUNT = 64
WEFT_COUNT = 64
WARP_WIDTH  = 0.055
WARP_THICK  = 0.02
WEFT_WIDTH  = 0.06
WEFT_THICK  = 0.02

SPACING_WARP = 0.09
SPACING_WEFT = 0.06
CRIMP_AMP   = 0.0032
AUTO_CLEARANCE   = True
CLEARANCE_MARGIN = 0.05
STEPS_PER_CELL = 6
BEVEL_RES      = 3
RESOLUTION_U   = 24

WARP_COLOR = (0.08, 0.08, 0.08, 1.0)
WEFT_COLOR = (0.85, 0.85, 0.85, 1.0)

# ========================= (렌더 엔진/라이트) ====================
USE_CYCLES   = True
SAMPLES      = 256
USE_DENOISER = True
CYCLES_DEVICE_TYPE = 'CUDA'

ADD_CAMERA_LIGHT = True
LIGHT_ENERGY      = 250.0
LIGHT_HEIGHT      = 1.0
LIGHT_SIZE_X_MULT = 1.2
LIGHT_SIZE_Y_MULT = 1.2

# ========================= (카메라) ==============================
OBLIQUE_VIEW       = True
CAMERA_PERSPECTIVE = True
CAMERA_ELEV_DEG    = 60.0
CAMERA_AZIMUTH_DEG = 0.0
CAMERA_RADIUS_MULT = 2.2
CAMERA_LENS_MM     = 50.0

# ▶ WeaveCam 절대 위치/회전(기존 요구대로 상단에 유지)
WEAVECAM_LOCATION = (0.0, -4.3352, 7.6820)   # 원하는 값으로 변경
WEAVECAM_LOOK_AT_TARGET = True               # target(OBJECT_OFFSET) 바라보도록
WEAVECAM_ROTATION_EULER_DEG = None          # (rx, ry, rz) 도 단위 직접 지정 시 사용

# ========================= (텍스처/UV/오프셋) ====================
UV_PROJECT_FROM_TOP = True
USE_IMAGE_TEXTURES  = True
TEX_REPEAT_WARP_UV  = (1.0, 1.0)
TEX_REPEAT_WEFT_UV  = (1.0, 1.0)
ROTATE_WEFT_90_DEG  = False
OBJECT_OFFSET       = (0.0, 0.1, 0.0)

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
    prof_warp=bpy.data.objects.get("Profile_Warp_Rect") or make_rect_profile("Profile_Warp_Rect", warp_width, warp_thick)
    prof_weft=bpy.data.objects.get("Profile_Weft_Rect") or make_rect_profile("Profile_Weft_Rect", weft_width, weft_thick)
    try:
        prof_warp.hide_set(True); prof_weft.hide_set(True)
        prof_warp.hide_render=True; prof_weft.hide_render=True
    except Exception: pass

    mat_warp=make_material("Mat_Warp", WARP_COLOR)
    mat_weft=make_material("Mat_Weft", WEFT_COLOR)

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

# ---------- 멀티 원단 패턴 적용 ----------
def build_material_bank(paths, prefix_w="Mat_Warp_IMG_", prefix_we="Mat_Weft_IMG_",
                        repeat_w=(1.0,1.0), repeat_we=(1.0,1.0), rotate_we_deg=0.0):
    warp_mats=[]; weft_mats=[]
    for idx, p in enumerate(paths):
        if p is None: continue
        mw = make_image_material(f"{prefix_w}{idx}", p, repeat=repeat_w, rotate_deg=0.0)
        me = make_image_material(f"{prefix_we}{idx}", p, repeat=repeat_we, rotate_deg=rotate_we_deg)
        warp_mats.append(mw); weft_mats.append(me)
    return warp_mats, weft_mats

def apply_pattern_materials(warp_objs, weft_objs, projector_obj,
                            warp_mats, weft_mats,
                            warp_index_fn, weft_index_fn):
    # 곡선을 한 번만 메쉬로 변환
    objs = [o for o in (warp_objs + weft_objs) if o is not None]
    convert_to_mesh_if_needed(objs)

    # UV Project 보장
    for o in warp_objs:
        if o and o.type=='MESH':
            ensure_uv_project_modifier(o, projector_obj, uv_name="UVMap")
    for o in weft_objs:
        if o and o.type=='MESH':
            ensure_uv_project_modifier(o, projector_obj, uv_name="UVMap")

    # 인덱스 함수에 따라 재질 할당
    for i, o in enumerate(warp_objs):
        if not o or o.type!='MESH': continue
        idx = warp_index_fn(i) % max(1, len(warp_mats))
        mat = warp_mats[idx]
        if o.data.materials: o.data.materials[0]=mat
        else: o.data.materials.append(mat)

    for j, o in enumerate(weft_objs):
        if not o or o.type!='MESH': continue
        idx = weft_index_fn(j) % max(1, len(weft_mats))
        mat = weft_mats[idx]
        if o.data.materials: o.data.materials[0]=mat
        else: o.data.materials.append(mat)

# ---------- 메인 ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 초기 정리
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH'); bpy.ops.object.delete()

    setup_cycles_engine(USE_CYCLES, SAMPLES, enable_gpu=True, device_type=CYCLES_DEVICE_TYPE, use_denoiser=USE_DENOISER)

    # 직물 생성
    result = generate_plain_weave(
        warp_count=WARP_COUNT, weft_count=WEFT_COUNT,
        warp_width=WARP_WIDTH, weft_width=WEFT_WIDTH,
        warp_thick=WARP_THICK, weft_thick=WEFT_THICK,
        spacing_warp=SPACING_WARP, spacing_weft=SPACING_WEFT,
        crimp_amp=CRIMP_AMP,
        auto_clearance=AUTO_CLEARANCE, clearance_margin=CLEARANCE_MARGIN,
        steps_per_cell=STEPS_PER_CELL, bevel_res=BEVEL_RES, resolution_u=RESOLUTION_U,
        warp_color=WARP_COLOR, weft_color=WEFT_COLOR
    )

    # 위치 오프셋
    all_objs = result["warp_objs"] + result["weft_objs"]
    for o in all_objs:
        if o: o.location = (o.location.x + OBJECT_OFFSET[0],
                            o.location.y + OBJECT_OFFSET[1],
                            o.location.z + OBJECT_OFFSET[2])
    target = OBJECT_OFFSET

    # 카메라/라이트
    cam, _ = setup_camera_and_top_light(ADD_CAMERA_LIGHT, result["total_w"], result["total_h"], target)
    bpy.context.scene.camera = cam

    # UV 프로젝터
    projector = get_or_create_uv_projector(result["total_w"], result["total_h"], target) if UV_PROJECT_FROM_TOP else cam

    # 원단 재질 뱅크
    rotate_we_deg = 90.0 if ROTATE_WEFT_90_DEG else 0.0
    warp_mats_all, weft_mats_all = build_material_bank(
        FABRIC_PATHS,
        repeat_w=TEX_REPEAT_WARP_UV,
        repeat_we=TEX_REPEAT_WEFT_UV,
        rotate_we_deg=rotate_we_deg
    )

    # ===== 시나리오 정의 및 순차 실행 =====
    scenarios = [
        # 1) 하나의 원단으로 경사/위사 동일
        ("1_one_fabric",           1, lambda i: 0,      lambda j: 0),
        # 2) 두 개 원단: 경사=0번, 위사=1번
        ("2_two_fabrics",          2, lambda i: 0,      lambda j: 1),
        # 3) 세 개 원단 번갈아: 경사 i%3, 위사 j%3
        ("3_three_fabrics_alt",    3, lambda i: i % 3,  lambda j: j % 3),
        # 4) 네 개 원단 번갈아: 경사 i%4, 위사 j%4
        ("4_four_fabrics_alt",     4, lambda i: i % 4,  lambda j: j % 4),
    ]

    # 3D뷰 카메라 시점 전환(편의)
    scr=bpy.context.screen
    if scr:
        for area in scr.areas:
            if area.type=='VIEW_3D':
                for space in area.spaces:
                    if space.type=='VIEW_3D':
                        space.region_3d.view_perspective='CAMERA'; break

    for name, need_n, warp_idx_fn, weft_idx_fn in scenarios:
        if need_n > len(FABRIC_PATHS):
            print(f"[WARN] {name}: 필요한 원단 {need_n}개 > 제공 {len(FABRIC_PATHS)}개. 건너뜀.")
            continue

        # 이번 시나리오에서 사용할 재질 세트 추출(앞에서 need_n개만)
        warp_mats = warp_mats_all[:need_n]
        weft_mats = weft_mats_all[:need_n]

        # 패턴 적용
        apply_pattern_materials(result["warp_objs"], result["weft_objs"], projector,
                                warp_mats, weft_mats, warp_idx_fn, weft_idx_fn)

        # 렌더
        if DO_RENDER:
            out_path = os.path.join(OUTPUT_DIR, f"swatch_{name}.png")
            bpy.context.scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            print(f"[INFO] Render saved: {out_path}")
        else:
            print(f"[INFO] Applied pattern: {name} (미렌더)")

    print(f"[INFO] total size (m): {result['total_w']:.4f} × {result['total_h']:.4f}")
    print(f"[INFO] WeaveCam loc: {tuple(round(v,4) for v in bpy.data.objects['WeaveCam'].location)}")

if __name__=="__main__":
    main()
