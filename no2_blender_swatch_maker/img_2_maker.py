import bpy
import math
from math import sin, pi
from mathutils import Matrix, Vector

# =========================
DO_RENDER   = True                   # 렌더 실행 여부
OUTPUT_PATH = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no2_blender_swatch_maker\render_output\2img\rendered_2_tshirt2.png"  # 저장 경로

WARP_IMAGE_PATH = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\cropped_tshirt\9a67ffca-6ad4-40f8-8224-5afc44bb4022_crop1.png"  # 경사 텍스처
WEFT_IMAGE_PATH = r"C:\Users\_idal\PycharmProjects\Cloth_AI\data\fabric\fabric_0035_roughness_4k.jpg"  # 위사 텍스처
# =========================
WARP_COUNT = 64                      # 경사(세로 실) 개수
WEFT_COUNT = 64                      # 위사(가로 실) 개수
WARP_WIDTH  = 0.055                  # 경사 폭(m)
WARP_THICK  = 0.02                   # 경사 두께(m)
WEFT_WIDTH  = 0.06                   # 위사 폭(m)
WEFT_THICK  = 0.02                   # 위사 두께(m)

SPACING_WARP = 0.09                  # 경사 피치(센터-센터, m)
SPACING_WEFT = 0.06                  # 위사 피치(센터-센터, m)
CRIMP_AMP   = 0.0032                 # 교차부 굴곡 진폭(m)
AUTO_CLEARANCE   = True              # 간섭 방지 자동 보정
CLEARANCE_MARGIN = 0.05              # 간섭 보정 여유율(비율)
STEPS_PER_CELL = 6                   # 한 셀(교차 간격)당 곡선 분할 수
BEVEL_RES      = 3                   # 커브 베벨 해상도
RESOLUTION_U   = 24                  # 커브 자체 해상도
WARP_COLOR = (0.08, 0.08, 0.08, 1.0) # 경사 기본 색상(sRGB)
WEFT_COLOR = (0.85, 0.85, 0.85, 1.0) # 위사 기본 색상(sRGB)
USE_CYCLES   = True                  # Cycles 사용 여부(Eevee 사용 시 False)
SAMPLES      = 256                   # Cycles 샘플 수
USE_DENOISER = True                  # 노이즈 제거 사용
CYCLES_DEVICE_TYPE = 'CUDA'          # 'CUDA' | 'OPTIX' | 'HIP'
ADD_CAMERA_LIGHT = True              # 상단 단일 AREA 라이트 추가
LIGHT_ENERGY      = 200.0            # 라이트 광량(W)
LIGHT_HEIGHT      = 1.0              # 라이트 높이(m)
LIGHT_SIZE_X_MULT = 1.2              # 라이트 X 크기 배율(시료 폭 기준)
LIGHT_SIZE_Y_MULT = 1.2              # 라이트 Y 크기 배율(시료 높이 기준)

OBLIQUE_VIEW       = True            # 비스듬한 시점(True) / 탑다운(False)
CAMERA_PERSPECTIVE = False            # 원근(True) / 직교(False)
CAMERA_ELEV_DEG    = 60.0            # (수동 위치를 쓰지 않을 때) 기준 고도
CAMERA_AZIMUTH_DEG = 0.0             # (수동 위치를 쓰지 않을 때) 기준 방위각
CAMERA_RADIUS_MULT = 2.2             # (수동 위치를 쓰지 않을 때) 거리 배수
CAMERA_LENS_MM     = 50.0            # 카메라 렌즈(mm, 원근일 때)

# ▶ WeaveCam 수동 위치/회전 지정 (원하는 값으로 바꿔서 사용)
WEAVECAM_LOCATION = (0.0, -5.5, 7.6820)   # 절대 좌표 (기본값: 이전 계산 결과와 유사)
WEAVECAM_LOOK_AT_TARGET = True               # True면 target(OBJECT_OFFSET)을 바라보도록 회전 계산
WEAVECAM_ROTATION_EULER_DEG = None           # 예: (60.0, 0.0, 0.0) 주면 look-at 대신 이 회전 사용

UV_PROJECT_FROM_TOP = True           # 텍스처 UV 투영을 탑다운 카메라로 수행
USE_IMAGE_TEXTURES = True            # 이미지 텍스처 사용
TEX_REPEAT_WARP_UV = (1.0, 1.0)      # 경사 텍스처 반복(U, V)
TEX_REPEAT_WEFT_UV = (1.0, 1.0)      # 위사 텍스처 반복(U, V)
ROTATE_WEFT_90_DEG = False           # 위사 텍스처 90° 회전
OBJECT_OFFSET = (0.0, 0.1, 0.0)      # 생성물 위치 오프셋(X, Y, Z) - Y+로 살짝 올림
#################################################################################


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
            try:
                sock.default_value = value
            except Exception:
                pass
            return True
    return False

def make_material(name, rgba):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (400, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (150, 0)
    _safe_set_input(bsdf, "Base Color", rgba)
    _safe_set_input(bsdf, "Roughness", 0.8)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def poly_curve(name, points):
    cu = bpy.data.curves.new(name=name, type='CURVE')
    cu.dimensions = '3D'
    cu.fill_mode = 'FULL'
    spline = cu.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for i, (x, y, z) in enumerate(points):
        spline.points[i].co = (x, y, z, 1.0)
    obj = bpy.data.objects.new(name, cu)
    return obj

def make_rect_profile(name, width, thickness):
    cu = bpy.data.curves.new(name=name, type='CURVE')
    cu.dimensions = '2D'
    cu.fill_mode = 'BOTH'
    sp = cu.splines.new('POLY')
    sp.use_cyclic_u = True
    hw = float(width) * 0.5
    ht = float(thickness) * 0.5
    pts = [(-hw, -ht, 0.0), (hw, -ht, 0.0), (hw, ht, 0.0), (-hw, ht, 0.0)]
    sp.points.add(len(pts) - 1)
    for i, (x, y, z) in enumerate(pts):
        sp.points[i].co = (x, y, z, 1.0)
    obj = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def build_axis_curve(axis, fixed_pos, count_other, spacing_along, idx_self, mat,
                     width, thickness, steps_per_cell=6, crimp_amp=0.0032,
                     bevel_profile_obj=None):
    half_len = (count_other * spacing_along) * 0.5
    total_steps = count_other * steps_per_cell
    pts = []
    for s in range(total_steps + 1):
        t = s / total_steps * (count_other * spacing_along)
        j = min(int(t // spacing_along), count_other - 1)
        u = (t - j * spacing_along) / spacing_along
        if axis == 'warp':
            x = fixed_pos
            y = -half_len + t
            over = ((idx_self + j) % 2) == 0
            sign = +1 if over else -1
            z = sign * crimp_amp * sin(pi * u)
        else:
            x = -half_len + t
            y = fixed_pos
            over = ((j + idx_self) % 2) == 0
            sign = -1 if over else +1
            z = sign * crimp_amp * sin(pi * u)
        pts.append((x, y, z))
    name = f"{'Warp' if axis=='warp' else 'Weft'}_{idx_self:02d}"
    obj = poly_curve(name, pts)
    obj.data.bevel_mode = 'OBJECT'
    if bevel_profile_obj is not None:
        obj.data.bevel_object = bevel_profile_obj
    obj.data.bevel_factor_start = 0.0
    obj.data.bevel_factor_end = 1.0
    obj.data.bevel_resolution = BEVEL_RES
    obj.data.resolution_u = RESOLUTION_U
    try:
        obj.data.twist_mode = 'Z_UP'
    except Exception:
        pass
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)
    return obj

def auto_clearance_adjust_rect(warp_width, weft_width, spacing_warp, spacing_weft,
                               crimp_amp, warp_thick, weft_thick, margin):
    min_warp_pitch = float(warp_width) * (1.0 + margin)
    min_weft_pitch = float(weft_width) * (1.0 + margin)
    if spacing_warp < min_warp_pitch: spacing_warp = min_warp_pitch
    if spacing_weft < min_weft_pitch: spacing_weft = min_weft_pitch
    min_center_gap = (float(warp_thick) + float(weft_thick)) * (1.0 + margin)
    if 2.0 * crimp_amp < min_center_gap:
        crimp_amp = 0.5 * min_center_gap
    return spacing_warp, spacing_weft, crimp_amp

def setup_cycles_engine(use_cycles=True, samples=256, enable_gpu=True, device_type='CUDA',
                        use_denoiser=True):
    scene = bpy.context.scene
    if use_cycles:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = samples
        scene.render.film_transparent = True
        scene.view_settings.exposure = 0.0
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.adaptive_threshold = 0.01
        if hasattr(scene.cycles, "use_light_tree"):
            scene.cycles.use_light_tree = True
        if use_denoiser:
            try:
                if hasattr(scene.cycles, "denoiser"):
                    scene.cycles.denoiser = 'OPTIX'
                bpy.context.view_layer.cycles.use_denoising = True
            except Exception:
                try:
                    if hasattr(scene.cycles, "denoiser"):
                        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
                    bpy.context.view_layer.cycles.use_denoising = True
                except Exception:
                    pass
        if enable_gpu:
            try:
                prefs = bpy.context.preferences
                cprefs = prefs.addons['cycles'].preferences
                cprefs.use_persistent_data = True
                cprefs.compute_device_type = device_type
                cprefs.get_devices()
                for dev in cprefs.devices:
                    dev.use = True
                scene.cycles.device = 'GPU'
            except Exception as e:
                print(f"[WARN] Cycles GPU 설정 실패: {e}")
    else:
        scene.render.engine = 'BLENDER_EEVEE'

def make_image_material(name, image_path, repeat=(1.0, 1.0), rotate_deg=0.0, roughness=1.0):
    img = None
    try:
        img = bpy.data.images.load(image_path, check_existing=True)
    except Exception as e:
        print(f"[WARN] 이미지 로드 실패: {image_path} ({e})")
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (1000, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (760, 0)
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.location = (540, 0)
    mapn = nt.nodes.new("ShaderNodeMapping"); mapn.location = (320, 0)
    tco = nt.nodes.new("ShaderNodeTexCoord"); tco.location = (100, 0)
    if img is not None:
        tex.image = img
        tex.extension = 'REPEAT'
        try:
            tex.image.colorspace_settings.name = "sRGB"
        except Exception:
            pass
    bsdf.inputs["Roughness"].default_value = roughness
    mapn.inputs["Scale"].default_value[0] = max(1e-6, float(repeat[0]))
    mapn.inputs["Scale"].default_value[1] = max(1e-6, float(repeat[1]))
    mapn.inputs["Rotation"].default_value[2] = math.radians(float(rotate_deg))
    nt.links.new(tco.outputs.get("UV") or tco.outputs["Generated"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def ensure_uv_layer(obj, name="UVMap"):
    if obj.type != 'MESH':
        return
    names = [l.name for l in obj.data.uv_layers]
    if name not in names:
        obj.data.uv_layers.new(name=name)
    obj.data.uv_layers.active = obj.data.uv_layers.get(name)

def ensure_uv_project_modifier(obj, projector_obj, uv_name="UVMap"):
    if obj.type != 'MESH':
        return
    ensure_uv_layer(obj, uv_name)
    mod = obj.modifiers.get("UVProject") or obj.modifiers.new("UVProject", type='UV_PROJECT')
    if len(mod.projectors) < 1:
        mod.projectors_add()
    mod.projectors[0].object = projector_obj
    mod.uv_layer = uv_name
    mod.aspect_x = 1.0
    mod.aspect_y = 1.0

def apply_images_to_warp_weft(warp_objs, weft_objs, projector_obj,
                              warp_img_path, weft_img_path,
                              warp_repeat=(1.0,1.0), weft_repeat=(1.0,1.0),
                              rotate_weft_90=False):
    bpy.ops.object.select_all(action='DESELECT')
    objs = [o for o in (warp_objs + weft_objs) if o is not None]
    if not objs:
        return
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.convert(target='MESH')
    mat_warp_img = make_image_material("Mat_Warp_IMG", warp_img_path, repeat=warp_repeat, rotate_deg=0.0)
    mat_weft_img = make_image_material("Mat_Weft_IMG", weft_img_path, repeat=weft_repeat,
                                       rotate_deg=(90.0 if rotate_weft_90 else 0.0))
    for obj in warp_objs:
        if obj and obj.type == 'MESH':
            ensure_uv_project_modifier(obj, projector_obj, uv_name="UVMap")
            if obj.data.materials: obj.data.materials[0] = mat_warp_img
            else: obj.data.materials.append(mat_warp_img)
    for obj in weft_objs:
        if obj and obj.type == 'MESH':
            ensure_uv_project_modifier(obj, projector_obj, uv_name="UVMap")
            if obj.data.materials: obj.data.materials[0] = mat_weft_img
            else: obj.data.materials.append(mat_weft_img)

def setup_camera_and_top_light(add_camera_light=True, total_w=1.0, total_h=1.0, target=(0.0,0.0,0.0)):
    scene = bpy.context.scene
    smax = max(float(total_w), float(total_h))

    # --- WeaveCam 생성/준비 ---
    cam = bpy.data.objects.get("WeaveCam")
    if cam is None:
        cam_data = bpy.data.cameras.new("WeaveCam")
        cam = bpy.data.objects.new("WeaveCam", cam_data)
        scene.collection.objects.link(cam)
    cam.rotation_mode = 'XYZ'
    if hasattr(cam.data, "shift_x"): cam.data.shift_x = 0.0
    if hasattr(cam.data, "shift_y"): cam.data.shift_y = 0.0
    cam.data.clip_start = 0.001
    cam.data.clip_end   = 100.0

    # 카메라 타입/렌즈
    cam.data.type = 'PERSP' if (OBLIQUE_VIEW and CAMERA_PERSPECTIVE) else 'ORTHO'
    if cam.data.type == 'ORTHO':
        cam.data.ortho_scale = smax * 1.15
    if hasattr(cam.data, "lens"):
        cam.data.lens = float(CAMERA_LENS_MM)

    # 위치 설정: 우선 수동 위치 사용, 없으면 기존 계산식 사용
    if WEAVECAM_LOCATION is not None:
        cam.location = (float(WEAVECAM_LOCATION[0]), float(WEAVECAM_LOCATION[1]), float(WEAVECAM_LOCATION[2]))
    else:
        elev = math.radians(float(CAMERA_ELEV_DEG))
        az   = math.radians(float(CAMERA_AZIMUTH_DEG))
        r    = max(0.25, float(CAMERA_RADIUS_MULT) * smax) * 0.7
        base_x = target[0] + r * math.sin(az) * math.cos(elev)
        base_y = target[1] - r * math.cos(az) * math.cos(elev)
        base_z = target[2] + r * math.sin(elev)
        cam.location = (base_x, base_y, base_z)

    # 회전 설정: 명시 회전 > look-at(target) > 유지
    if WEAVECAM_ROTATION_EULER_DEG is not None:
        rx, ry, rz = [math.radians(float(a)) for a in WEAVECAM_ROTATION_EULER_DEG]
        cam.rotation_euler = (rx, ry, rz)
    elif WEAVECAM_LOOK_AT_TARGET:
        forward = (Vector(target) - Vector(cam.location)).normalized()
        world_up = Vector((0.0, 0.0, 1.0))
        right = forward.cross(world_up).normalized()
        if right.length < 1e-6:
            world_up = Vector((0.0, 1.0, 0.0))
            right = forward.cross(world_up).normalized()
        up = right.cross(forward).normalized()
        rot_mat = Matrix(((right.x,  up.x,  -forward.x),
                          (right.y,  up.y,  -forward.y),
                          (right.z,  up.z,  -forward.z)))
        cam.rotation_euler = rot_mat.to_euler('XYZ')

    # --- 라이트 설정 ---
    light = None
    if add_camera_light:
        keep_name = "Light"
        for obj in list(scene.objects):
            if obj.type == 'LIGHT' and obj.name != keep_name:
                bpy.data.objects.remove(obj, do_unlink=True)
        light = bpy.data.objects.get(keep_name)
        if light is None:
            light_data = bpy.data.lights.new(name=keep_name, type='AREA')
            light = bpy.data.objects.new(name=keep_name, object_data=light_data)
            scene.collection.objects.link(light)
        light.data.type = 'AREA'
        light.data.shape = 'RECTANGLE'
        light.data.size   = max(smax * LIGHT_SIZE_X_MULT, 0.05)
        light.data.size_y = max(smax * LIGHT_SIZE_Y_MULT, 0.05)
        light.rotation_mode = 'XYZ'
        light.rotation_euler = (0.0, 0.0, 0.0)
        light.location = (float(target[0]), float(target[1]), float(LIGHT_HEIGHT))
        light.data.energy = max(float(LIGHT_ENERGY), 0.0)
        try:
            light.data.use_contact_shadow = True
        except Exception:
            pass

    # 렌더 메인 카메라 고정
    scene.camera = cam
    return cam, light

def get_or_create_uv_projector(total_w, total_h, target=(0.0,0.0,0.0), name="UVProjectorTop"):
    smax = max(float(total_w), float(total_h))
    cam = bpy.data.objects.get(name)
    if cam is None:
        cam_data = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = smax * 1.5
    cam.location = (float(target[0]), float(target[1]), float(target[2]) + 1.0)
    cam.rotation_mode = 'XYZ'
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.data.clip_start = 0.001
    cam.data.clip_end = 100.0
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
    BEVEL_RES = bevel_res
    RESOLUTION_U = resolution_u
    STEPS_PER_CELL = steps_per_cell
    if auto_clearance:
        spacing_warp, spacing_weft, crimp_amp = auto_clearance_adjust_rect(
            warp_width, weft_width, spacing_warp, spacing_weft,
            crimp_amp, warp_thick, weft_thick, clearance_margin
        )
    col = clean_collection("PlainWeave_Swatch")
    prof_warp = bpy.data.objects.get("Profile_Warp_Rect")
    if prof_warp is None:
        prof_warp = make_rect_profile("Profile_Warp_Rect", warp_width, warp_thick)
    prof_weft = bpy.data.objects.get("Profile_Weft_Rect")
    if prof_weft is None:
        prof_weft = make_rect_profile("Profile_Weft_Rect", weft_width, weft_thick)
    try:
        prof_warp.hide_set(True); prof_weft.hide_set(True)
        prof_warp.hide_render = True; prof_weft.hide_render = True
    except Exception:
        pass
    mat_warp = make_material("Mat_Warp", WARP_COLOR)
    mat_weft = make_material("Mat_Weft", WEFT_COLOR)
    total_w = warp_count * spacing_warp
    total_h = weft_count * spacing_weft
    x0 = -0.5 * total_w + spacing_warp * 0.5
    y0 = -0.5 * total_h + spacing_weft * 0.5
    warp_objs, weft_objs = [], []
    for i in range(warp_count):
        x = x0 + i * spacing_warp
        obj = build_axis_curve(
            axis='warp', fixed_pos=x,
            count_other=weft_count, spacing_along=spacing_weft,
            idx_self=i, mat=mat_warp,
            width=warp_width, thickness=warp_thick,
            steps_per_cell=steps_per_cell, crimp_amp=crimp_amp,
            bevel_profile_obj=prof_warp
        ); col.objects.link(obj); warp_objs.append(obj)
    for j in range(weft_count):
        y = y0 + j * spacing_weft
        obj = build_axis_curve(
            axis='weft', fixed_pos=y,
            count_other=warp_count, spacing_along=spacing_warp,
            idx_self=j, mat=mat_weft,
            width=weft_width, thickness=weft_thick,
            steps_per_cell=steps_per_cell, crimp_amp=crimp_amp,
            bevel_profile_obj=prof_weft
        ); col.objects.link(obj); weft_objs.append(obj)
    return {"collection": col, "total_w": total_w, "total_h": total_h,
            "warp_objs": warp_objs, "weft_objs": weft_objs}

def apply_object_offset(objs, offset=(0.0,0.0,0.0)):
    ox, oy, oz = float(offset[0]), float(offset[1]), float(offset[2])
    if abs(ox)+abs(oy)+abs(oz) < 1e-12:
        return
    for o in objs:
        if o: o.location = (o.location.x + ox, o.location.y + oy, o.location.z + oz)

def main():
    # 초기 정리
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # 렌더 엔진/장치 설정
    setup_cycles_engine(
        use_cycles=USE_CYCLES,
        samples=SAMPLES,
        enable_gpu=True,
        device_type=CYCLES_DEVICE_TYPE,
        use_denoiser=USE_DENOISER
    )

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

    # 오브젝트 오프셋 적용
    all_curve_objs = result["warp_objs"] + result["weft_objs"]
    apply_object_offset(all_curve_objs, OBJECT_OFFSET)
    target = OBJECT_OFFSET

    # 카메라/라이트 세팅 (scene.camera 고정 포함)
    cam, _ = setup_camera_and_top_light(
        add_camera_light=ADD_CAMERA_LIGHT,
        total_w=result["total_w"], total_h=result["total_h"], target=target
    )
    bpy.context.scene.camera = cam  # 안전장치(중복 지정)

    # 텍스처 적용 (필요 시)
    if USE_IMAGE_TEXTURES:
        if UV_PROJECT_FROM_TOP:
            projector = get_or_create_uv_projector(result["total_w"], result["total_h"], target=target)
        else:
            projector = bpy.data.objects.get("WeaveCam")
            if projector is None:
                raise RuntimeError("WeaveCam 카메라가 없습니다.")
        apply_images_to_warp_weft(
            result["warp_objs"], result["weft_objs"], projector,
            warp_img_path=WARP_IMAGE_PATH,
            weft_img_path=WEFT_IMAGE_PATH,
            warp_repeat=TEX_REPEAT_WARP_UV,
            weft_repeat=TEX_REPEAT_WEFT_UV,
            rotate_weft_90=ROTATE_WEFT_90_DEG
        )

    # 3D뷰를 카메라 시점으로 전환(편의)
    scr = bpy.context.screen
    if scr:
        for area in scr.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.region_3d.view_perspective = 'CAMERA'
                        break

    print(f"[INFO] total size (m): {result['total_w']:.4f} × {result['total_h']:.4f}")
    print(f"[INFO] WeaveCam loc: {tuple(round(v, 4) for v in bpy.data.objects['WeaveCam'].location)}")
    print(f"[INFO] WeaveCam rot(deg): {tuple(round(math.degrees(a), 2) for a in bpy.data.objects['WeaveCam'].rotation_euler)}")

if __name__ == "__main__":
    main()
    if DO_RENDER:
        scene = bpy.context.scene
        scene.render.filepath = OUTPUT_PATH
        bpy.ops.render.render(write_still=True)
        print(f"[INFO] Render saved to {scene.render.filepath}")
