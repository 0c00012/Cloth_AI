# Plain Weave Swatch Generator — 4이미지 번갈아 적용(경사/위사), UV Project from camera
# Blender 3.x / 4.x

import bpy
import math
from math import sin, pi
from mathutils import Matrix

# =========================
# 사용자 매개변수
# =========================
WARP_COUNT = 100
WEFT_COUNT = 100

WARP_WIDTH  = 0.05
WARP_THICK  = 0.02
WEFT_WIDTH  = 0.06
WEFT_THICK  = 0.02

SPACING_WARP = 0.09
SPACING_WEFT = 0.06
CRIMP_AMP    = 0.0032

AUTO_CLEARANCE   = True
CLEARANCE_MARGIN = 0.05

STEPS_PER_CELL = 6
BEVEL_RES      = 3
RESOLUTION_U   = 24

WARP_COLOR = (0.08, 0.08, 0.08, 1.0)
WEFT_COLOR = (0.85, 0.85, 0.85, 1.0)

USE_CYCLES       = True
SAMPLES          = 256
USE_DENOISER     = True
ADD_CAMERA_LIGHT = True
FORCE_TOPDOWN    = True
CYCLES_DEVICE_TYPE = 'CUDA'  # 'CUDA'|'OPTIX'|'HIP'

LIGHT_ENERGY      = 250.0
LIGHT_HEIGHT      = 1.0
LIGHT_SIZE_X_MULT = 1.2
LIGHT_SIZE_Y_MULT = 1.2

# ====== 원래 쓰던 2장(참고) ======
WARP_IMAGE_PATH = r"C:\Users\_idal\Desktop\texture_transferred.jpg"
WEFT_IMAGE_PATH = r"C:\Users\_idal\Downloads\fabric_0035_4k_Lbr8QW\fabric_0035_height_4k.png"

# ====== 새 옵션: 4장 번갈아 적용 ======
USE_IMAGE_TEXTURES      = True
USE_ALTERNATE_4_IMAGES  = True    # ← 이걸 True로 두면 4장 번갈아
ALTERNATE_IMAGE_PATHS   = [
    WARP_IMAGE_PATH,  # 이미지1 (경사 A)
    WEFT_IMAGE_PATH,  # 이미지2 (위사 A)
    r"C:\Users\_idal\Desktop\1.png",  # 이미지3 (경사 B)
    r"C:\Users\_idal\Desktop\tshirt\스크린샷 2025-08-15 121614.png",  # 이미지4 (위사 B)
]
# 반복(스케일)
TEX_REPEAT_ALL_UV = (1.0, 1.0)
# 위사 텍스처를 90° 회전해서 사용할지 (경사엔 회전 없음)
ROTATE_WEFT_90_DEG = False

# =========================
# 유틸리티
# =========================
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
    hw = float(width)*0.5; ht = float(thickness)*0.5
    pts = [(-hw,-ht,0.0),(hw,-ht,0.0),(hw,ht,0.0),(-hw,ht,0.0)]
    sp.points.add(len(pts)-1)
    for i,(x,y,z) in enumerate(pts):
        sp.points[i].co = (x,y,z,1.0)
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
            x = fixed_pos; y = -half_len + t
            over = ((idx_self + j) % 2) == 0
            sign = +1 if over else -1
            z = sign * sin(pi*u) * crimp_amp
        else:
            x = -half_len + t; y = fixed_pos
            over = ((j + idx_self) % 2) == 0
            sign = -1 if over else +1
            z = sign * sin(pi*u) * crimp_amp
        pts.append((x,y,z))

    name = f"{'Warp' if axis=='warp' else 'Weft'}_{idx_self:02d}"
    obj = poly_curve(name, pts)

    obj.data.bevel_mode = 'OBJECT'
    if bevel_profile_obj is not None:
        obj.data.bevel_object = bevel_profile_obj
    obj.data.bevel_factor_start = 0.0
    obj.data.bevel_factor_end   = 1.0
    obj.data.bevel_resolution   = BEVEL_RES
    obj.data.resolution_u       = RESOLUTION_U
    obj.data.twist_mode         = 'Z_UP'
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
    if 2.0*crimp_amp < min_center_gap:
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
                for dev in cprefs.devices: dev.use = True
                scene.cycles.device = 'GPU'
            except Exception as e:
                print(f"[WARN] Cycles GPU 설정 실패: {e}")
    else:
        scene.render.engine = 'BLENDER_EEVEE'

# ---------- 이미지 텍스처 & UV 투영(회전/스케일 처리) ----------
def make_image_material(name, image_path, repeat=(1.0, 1.0), rotate_deg=0.0):
    img = None
    try:
        img = bpy.data.images.load(image_path, check_existing=True)
    except Exception as e:
        print(f"[WARN] 이미지 로드 실패: {image_path} ({e})")

    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)

    out   = nt.nodes.new("ShaderNodeOutputMaterial"); out.location   = (1200,   0)
    bsdf  = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location  = (940,    0)
    tex   = nt.nodes.new("ShaderNodeTexImage");       tex.location   = (720,    0)
    tco   = nt.nodes.new("ShaderNodeTexCoord");       tco.location   = (-560,   0)
    sep   = nt.nodes.new("ShaderNodeSeparateXYZ");    sep.location   = (-360,   0)
    subx  = nt.nodes.new("ShaderNodeMath");           subx.location  = (-160,  80);  subx.operation = 'SUBTRACT'
    suby  = nt.nodes.new("ShaderNodeMath");           suby.location  = (-160, -80);  suby.operation = 'SUBTRACT'
    mulx  = nt.nodes.new("ShaderNodeMath");           mulx.location  = (  40,  80);  mulx.operation = 'MULTIPLY'
    muly  = nt.nodes.new("ShaderNodeMath");           muly.location  = (  40, -80);  muly.operation = 'MULTIPLY'
    xr_a  = nt.nodes.new("ShaderNodeMath");           xr_a.location  = ( 240,  80);  xr_a.operation = 'MULTIPLY'
    xr_b  = nt.nodes.new("ShaderNodeMath");           xr_b.location  = ( 240, -80);  xr_b.operation = 'MULTIPLY'
    xr    = nt.nodes.new("ShaderNodeMath");           xr.location    = ( 440,   0);  xr.operation   = 'SUBTRACT'
    yr_a  = nt.nodes.new("ShaderNodeMath");           yr_a.location  = ( 240, 160);  yr_a.operation = 'MULTIPLY'
    yr_b  = nt.nodes.new("ShaderNodeMath");           yr_b.location  = ( 240,-160);  yr_b.operation = 'MULTIPLY'
    yr    = nt.nodes.new("ShaderNodeMath");           yr.location    = ( 440, -140); yr.operation   = 'ADD'
    addx  = nt.nodes.new("ShaderNodeMath");           addx.location  = ( 600,  60);  addx.operation = 'ADD'
    addy  = nt.nodes.new("ShaderNodeMath");           addy.location  = ( 600,-100);  addy.operation = 'ADD'
    comb  = nt.nodes.new("ShaderNodeCombineXYZ");     comb.location  = ( 880, -120)
    v_cos = nt.nodes.new("ShaderNodeValue");          v_cos.location = (  40,-230)
    v_sin = nt.nodes.new("ShaderNodeValue");          v_sin.location = (  40,-300)
    v_zero= nt.nodes.new("ShaderNodeValue");          v_zero.location= ( 780,-220)

    if img is not None:
        tex.image = img
        tex.extension = 'REPEAT'
        try: tex.image.colorspace_settings.name = "sRGB"
        except Exception: pass

    bsdf.inputs["Roughness"].default_value = 1.0

    nt.links.new(tco.outputs.get("UV") or tco.outputs["Generated"], sep.inputs["Vector"])
    nt.links.new(sep.outputs["X"], subx.inputs[0]); subx.inputs[1].default_value = 0.5
    nt.links.new(sep.outputs["Y"], suby.inputs[0]); suby.inputs[1].default_value = 0.5

    sx = max(1e-6, float(repeat[0])); sy = max(1e-6, float(repeat[1]))
    nt.links.new(subx.outputs[0], mulx.inputs[0]); mulx.inputs[1].default_value = sx
    nt.links.new(suby.outputs[0], muly.inputs[0]); muly.inputs[1].default_value = sy

    angle_rad = math.radians(float(rotate_deg))
    v_cos.outputs[0].default_value = math.cos(angle_rad)
    v_sin.outputs[0].default_value = math.sin(angle_rad)

    nt.links.new(mulx.outputs[0], xr_a.inputs[0]); nt.links.new(v_cos.outputs[0], xr_a.inputs[1])
    nt.links.new(muly.outputs[0], xr_b.inputs[0]); nt.links.new(v_sin.outputs[0], xr_b.inputs[1])
    nt.links.new(xr_a.outputs[0], xr.inputs[0]);   nt.links.new(xr_b.outputs[0], xr.inputs[1])

    nt.links.new(mulx.outputs[0], yr_a.inputs[0]); nt.links.new(v_sin.outputs[0], yr_a.inputs[1])
    nt.links.new(muly.outputs[0], yr_b.inputs[0]); nt.links.new(v_cos.outputs[0], yr_b.inputs[1])
    nt.links.new(yr_a.outputs[0], yr.inputs[0]);   nt.links.new(yr_b.outputs[0], yr.inputs[1])

    addx.inputs[1].default_value = 0.5
    addy.inputs[1].default_value = 0.5
    nt.links.new(xr.outputs[0], addx.inputs[0])
    nt.links.new(yr.outputs[0], addy.inputs[0])

    v_zero.outputs[0].default_value = 0.0
    nt.links.new(addx.outputs[0], comb.inputs["X"])
    nt.links.new(addy.outputs[0], comb.inputs["Y"])
    nt.links.new(v_zero.outputs[0], comb.inputs["Z"])

    nt.links.new(comb.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def ensure_uv_layer(obj, name="UVMap"):
    if obj.type != 'MESH': return
    names = [l.name for l in obj.data.uv_layers]
    if name not in names:
        obj.data.uv_layers.new(name=name)
    obj.data.uv_layers.active = obj.data.uv_layers.get(name)

def ensure_uv_project_modifier(obj, projector_obj, uv_name="UVMap"):
    if obj.type != 'MESH': return
    ensure_uv_layer(obj, uv_name)
    mod = obj.modifiers.get("UVProject") or obj.modifiers.new("UVProject", type='UV_PROJECT')
    if len(mod.projectors) < 1: mod.projectors_add()
    mod.projectors[0].object = projector_obj
    mod.uv_layer = uv_name
    mod.aspect_x = 1.0; mod.aspect_y = 1.0

# --- 새 기능: 4장 이미지를 경사/위사에 번갈아 적용 ---
def apply_alternate4_to_weave(warp_objs, weft_objs, camera_obj, image_paths,
                              repeat=(1.0,1.0), rotate_weft_90=False):
    """
    warp: img[0], img[2] 번갈아
    weft: img[1], img[3] 번갈아
    """
    # 커브 → 메쉬
    bpy.ops.object.select_all(action='DESELECT')
    objs = [o for o in (warp_objs + weft_objs) if o is not None]
    if not objs: return
    for o in objs: o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.convert(target='MESH')

    # 4개 머티리얼(총 4개만 생성/재사용)
    paths = [image_paths[i] if i < len(image_paths) else None for i in range(4)]
    mat_warp_A = make_image_material("Mat_Warp_A", paths[0], repeat=repeat, rotate_deg=0.0)
    mat_weft_A = make_image_material("Mat_Weft_A", paths[1], repeat=repeat, rotate_deg=(90.0 if rotate_weft_90 else 0.0))
    mat_warp_B = make_image_material("Mat_Warp_B", paths[2], repeat=repeat, rotate_deg=0.0)
    mat_weft_B = make_image_material("Mat_Weft_B", paths[3], repeat=repeat, rotate_deg=(90.0 if rotate_weft_90 else 0.0))

    # 경사: A,B,A,B...
    for i, obj in enumerate(warp_objs):
        if obj and obj.type == 'MESH':
            ensure_uv_project_modifier(obj, camera_obj, uv_name="UVMap")
            mat = mat_warp_A if (i % 2 == 0) else mat_warp_B
            if obj.data.materials: obj.data.materials[0] = mat
            else: obj.data.materials.append(mat)

    # 위사: A,B,A,B...
    for j, obj in enumerate(weft_objs):
        if obj and obj.type == 'MESH':
            ensure_uv_project_modifier(obj, camera_obj, uv_name="UVMap")
            mat = mat_weft_A if (j % 2 == 0) else mat_weft_B
            if obj.data.materials: obj.data.materials[0] = mat
            else: obj.data.materials.append(mat)

# ---------- 카메라 & 상단 단일 라이트 ----------
def setup_camera_and_top_light(add_camera_light=True, total_w=1.0, total_h=1.0):
    scene = bpy.context.scene
    cam = bpy.data.objects.get("WeaveCam")
    if cam is None:
        cam_data = bpy.data.cameras.new("WeaveCam")
        cam = bpy.data.objects.new("WeaveCam", cam_data)
        scene.collection.objects.link(cam)

    cam.rotation_mode = 'XYZ'
    cam.rotation_euler = (0.0, 0.0, 0.0)
    cam.location = (0.0, 0.0, max(0.25, 0.6 * max(total_w, total_h))) if FORCE_TOPDOWN else (0.0, 0.0, 0.25)
    cam.matrix_world = Matrix.Translation(cam.location)
    if cam.parent: cam.parent = None
    if hasattr(cam.data, "shift_x"): cam.data.shift_x = 0.0
    if hasattr(cam.data, "shift_y"): cam.data.shift_y = 0.0

    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = max(total_w, total_h) * 1.5
    cam.data.clip_start = 0.001; cam.data.clip_end = 100.0
    scene.camera = cam

    if not add_camera_light: return
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
    sx = max(total_w * LIGHT_SIZE_X_MULT, 0.05)
    sy = max(total_h * LIGHT_SIZE_Y_MULT, 0.05)
    light.data.size = sx; light.data.size_y = sy
    light.rotation_mode = 'XYZ'
    light.rotation_euler = (0.0, 0.0, 0.0)
    light.location = (0.0, 0.0, LIGHT_HEIGHT)
    if light.parent: light.parent = None
    light.matrix_world = Matrix.Translation(light.location)
    light.data.energy = max(float(LIGHT_ENERGY), 0.0)
    if hasattr(light.data, "use_contact_shadow"):
        light.data.use_contact_shadow = True

def generate_plain_weave(
    warp_count, weft_count,
    warp_width, weft_width,
    warp_thick, weft_thick,
    spacing_warp, spacing_weft,
    crimp_amp,
    auto_clearance=True, clearance_margin=0.05,
    steps_per_cell=6, bevel_res=3, resolution_u=24,
    warp_color=(0.08, 0.08, 0.08, 1.0),
    weft_color=(0.85, 0.85, 0.85, 1.0),
    add_camera_light=True
):
    global BEVEL_RES, RESOLUTION_U, STEPS_PER_CELL
    BEVEL_RES = bevel_res; RESOLUTION_U = resolution_u; STEPS_PER_CELL = steps_per_cell

    if auto_clearance:
        spacing_warp, spacing_weft, crimp_amp = auto_clearance_adjust_rect(
            warp_width, weft_width, spacing_warp, spacing_weft,
            crimp_amp, warp_thick, weft_thick, clearance_margin
        )

    col = clean_collection("PlainWeave_Swatch")

    prof_warp = bpy.data.objects.get("Profile_Warp_Rect") or make_rect_profile("Profile_Warp_Rect", warp_width, warp_thick)
    prof_weft = bpy.data.objects.get("Profile_Weft_Rect") or make_rect_profile("Profile_Weft_Rect", weft_width, weft_thick)
    prof_warp.hide_set(True); prof_weft.hide_set(True)
    if hasattr(prof_warp, "hide_render"): prof_warp.hide_render = True
    if hasattr(prof_weft, "hide_render"): prof_weft.hide_render = True

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

    for obj in warp_objs + weft_objs: obj.select_set(True)
    if warp_objs: bpy.context.view_layer.objects.active = warp_objs[0]

    setup_camera_and_top_light(add_camera_light=add_camera_light, total_w=total_w, total_h=total_h)

    scr = bpy.context.screen
    if scr:
        for area in scr.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        r3d = space.region_3d
                        r3d.view_location = (0.0, 0.0, 0.0)
                        r3d.view_distance = 1.0
                        r3d.view_perspective = 'CAMERA'
                        break

    print("✅ Plain weave (RECT) generated:",
          f"{warp_count}×{weft_count}, warp_w={warp_width}, warp_t={warp_thick}, "
          f"weft_w={weft_width}, weft_t={weft_thick}, "
          f"pitch=({spacing_warp},{spacing_weft}), crimp={crimp_amp}")
    return {"total_w": total_w, "total_h": total_h, "warp_objs": warp_objs, "weft_objs": weft_objs}

# =========================
# 메인
# =========================
def main():
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    setup_cycles_engine(
        use_cycles=USE_CYCLES, samples=SAMPLES,
        enable_gpu=True, device_type=CYCLES_DEVICE_TYPE,
        use_denoiser=USE_DENOISER
    )

    result = generate_plain_weave(
        warp_count=WARP_COUNT, weft_count=WEFT_COUNT,
        warp_width=WARP_WIDTH, weft_width=WEFT_WIDTH,
        warp_thick=WARP_THICK, weft_thick=WEFT_THICK,
        spacing_warp=SPACING_WARP, spacing_weft=SPACING_WEFT,
        crimp_amp=CRIMP_AMP,
        auto_clearance=AUTO_CLEARANCE, clearance_margin=CLEARANCE_MARGIN,
        steps_per_cell=STEPS_PER_CELL, bevel_res=BEVEL_RES, resolution_u=RESOLUTION_U,
        warp_color=WARP_COLOR, weft_color=WEFT_COLOR,
        add_camera_light=ADD_CAMERA_LIGHT
    )
    print(f"[INFO] total size (m): {result['total_w']:.4f} × {result['total_h']:.4f}")

    if USE_IMAGE_TEXTURES:
        cam = bpy.data.objects.get("WeaveCam")
        if cam is None:
            raise RuntimeError("WeaveCam 카메라가 없습니다.")
        if USE_ALTERNATE_4_IMAGES:
            apply_alternate4_to_weave(
                result["warp_objs"], result["weft_objs"], cam,
                image_paths=ALTERNATE_IMAGE_PATHS,
                repeat=TEX_REPEAT_ALL_UV,
                rotate_weft_90=ROTATE_WEFT_90_DEG
            )
        else:
            # (옵션) 기존 2장 방식으로 적용하고 싶을 때 여기로…
            pass

if __name__ == "__main__":
    main()
    #scene = bpy.context.scene
    #scene.render.filepath = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no2_blender_swatch_maker\render_output\4img\img_4_tshirt.png"  # 저장 경로
    #bpy.ops.render.render(write_still=True)
    #print(f"[INFO] Render saved to {scene.render.filepath}")
