import bpy
import sys
import os
import numpy as np
from math import sin, pi

# ==============================================================================
# [CONFIG]
# ==============================================================================

# Swatch geometry (Basis)
SWATCH_SIZE_CM = 10.0
SWATCH_SIZE_PX = 3755

# ★ [NEW] 랜덤 시드 고정
RANDOM_SEED = 42

# ★ [NEW] 경사/위사 개별 하이퍼파라미터 (단위: cm)
WARP_PITCH_CM = 0.5  # 경사 간격
WARP_WIDTH_CM = 0.35  # 경사 폭
WARP_THICK_CM = 0.15  # 경사 두께

WEFT_PITCH_CM = 0.8  # 위사 간격
WEFT_WIDTH_CM = 0.65  # 위사 폭
WEFT_THICK_CM = 0.20  # 위사 두께
WEFT_Z_OFFSET_M = 0.0003

# Renderer
SAMPLES = 2048
CM_EXPOSURE = 0.05

# World & Light
WORLD_STRENGTH = 0.15
WORLD_COLOR = (1.0, 1.0, 1.0, 1.0)
LIGHT_SIZE_MULT = 1.25
LIGHT_TOP_ENERGY_W = 22.0
LIGHT_TOP_HEIGHT_M = 1.5

# Crimp
CRIMP_RATIO = 0.32
CRIMP_SCALE = 0.7

# Details
STEPS_PER_CELL = 4
BEVEL_RES = 3

# Colors & Mask
WARP_WHITE = (1, 1, 1, 1)
WEFT_SIDE_COLOR = (0, 0, 0, 0)
TOP_MASK_LOW = 0.65
TOP_MASK_HIGH = 0.95

# ==============================================================================
# [DERIVED VALUES]
# ==============================================================================
CM2M = 0.01

# cm -> meter 변환
warp_pitch_m = WARP_PITCH_CM * CM2M
weft_pitch_m = WEFT_PITCH_CM * CM2M

warp_w = WARP_WIDTH_CM * CM2M
warp_t = WARP_THICK_CM * CM2M

weft_w = WEFT_WIDTH_CM * CM2M
weft_t = WEFT_THICK_CM * CM2M

# Crimp Amplitude (평균 피치 기준)
avg_pitch = (warp_pitch_m + weft_pitch_m) / 2
crimp_amp = avg_pitch * CRIMP_RATIO * CRIMP_SCALE


# ==============================================================================
# [FUNCTIONS]
# ==============================================================================

def set_random_seed(seed):
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        print(f"Set Random Seed: {seed}")


def nuke_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in (bpy.data.materials, bpy.data.images, bpy.data.textures, bpy.data.curves, bpy.data.lights,
              bpy.data.cameras, bpy.data.worlds):
        for b in c: c.remove(b)


def setup_renderer_and_world(filepath, cols, rows):
    scn = bpy.context.scene

    # 1. Renderer (Cycles)
    scn.render.engine = 'CYCLES'
    scn.cycles.samples = SAMPLES
    if hasattr(scn.cycles, 'use_adaptive_sampling'):
        scn.cycles.use_adaptive_sampling = True
        scn.cycles.adaptive_threshold = 0.01

    # GPU Setup
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        for dev_type in ('OPTIX', 'CUDA'):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                for d in prefs.devices: d.use = True
                scn.cycles.device = 'GPU'
                break
            except:
                pass
        scn.cycles.denoiser = 'OPTIX'
        bpy.context.view_layer.cycles.use_denoising = True
    except:
        pass

    # 2. Color Management
    vs = scn.view_settings
    try:
        vs.view_transform = 'Filmic'
    except:
        pass
    vs.exposure = CM_EXPOSURE
    vs.gamma = 1.0

    # 3. Resolution
    base_px = 376
    scale = 1
    scn.render.resolution_x = int(base_px * cols * scale)
    scn.render.resolution_y = int(base_px * rows * scale)

    # Output
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGBA'
    scn.render.film_transparent = True
    scn.render.filepath = filepath

    # 4. World
    world = bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    if not bg: bg = nt.nodes.new("ShaderNodeBackground")

    out = nt.nodes.get("World Output")
    if not out: out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    bg.inputs["Color"].default_value = WORLD_COLOR
    bg.inputs["Strength"].default_value = WORLD_STRENGTH


def make_rect_profile(name, width, thickness):
    cu = bpy.data.curves.new(name=name, type='CURVE')
    cu.dimensions = '2D'
    cu.fill_mode = 'BOTH'
    sp = cu.splines.new('POLY')
    sp.use_cyclic_u = True

    hw, ht = float(width) * 0.5, float(thickness) * 0.5
    pts = [(-hw, -ht, 0), (hw, -ht, 0), (hw, ht, 0), (-hw, ht, 0)]
    sp.points.add(len(pts) - 1)
    for i, (x, y, z) in enumerate(pts):
        sp.points[i].co = (x, y, z, 1.0)
    return bpy.data.objects.new(name, cu)


def build_thread_optimized(name, crossing_count, axis, idx, mat, prof, z_offset=0):
    if axis == 'warp':
        crossing_pitch = weft_pitch_m
    else:
        crossing_pitch = warp_pitch_m

    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    cu.fill_mode = 'FULL'
    spline = cu.splines.new('POLY')

    total_steps = crossing_count * STEPS_PER_CELL
    spline.points.add(total_steps)

    total_len_m = crossing_count * crossing_pitch
    half_len = total_len_m / 2

    # Coordinates
    t = np.linspace(0, total_len_m, total_steps + 1)

    j = np.minimum((t // crossing_pitch).astype(int), crossing_count - 1)
    u = (t - j * crossing_pitch) / crossing_pitch

    if axis == 'warp':
        x = np.zeros_like(t)
        y = -half_len + t
        phase = (idx + j) % 2
        sign = np.where(phase == 0, 1, -1)
        z = sign * crimp_amp * np.sin(pi * u)
    else:
        x = -half_len + t
        y = np.zeros_like(t)
        phase = (j + idx) % 2
        sign = np.where(phase == 0, -1, 1)
        z = sign * crimp_amp * np.sin(pi * u) + z_offset

    points = np.empty(len(t) * 4, dtype=np.float32)
    points[0::4] = x
    points[1::4] = y
    points[2::4] = z
    points[3::4] = 1.0
    spline.points.foreach_set('co', points)

    obj = bpy.data.objects.new(name, cu)
    obj.data.bevel_mode = 'OBJECT'
    obj.data.bevel_object = prof
    obj.data.bevel_resolution = BEVEL_RES
    obj.data.use_fill_caps = True
    obj.data.materials.append(mat)

    return obj


def setup_materials_reference_style(tex_path, total_w_m, total_h_m):
    # Warp White
    m_warp = bpy.data.materials.new("Mat_Warp_White")
    m_warp.use_nodes = True
    nt = m_warp.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bs.inputs["Base Color"].default_value = WARP_WHITE
    bs.inputs["Roughness"].default_value = 0.8
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])

    # Weft Texture
    m_weft = bpy.data.materials.new("Mat_Weft_Texture")
    m_weft.use_nodes = True
    nt = m_weft.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color1"].default_value = WEFT_SIDE_COLOR

    tex = nt.nodes.new("ShaderNodeTexImage")
    try:
        img = bpy.data.images.load(tex_path)
        img.colorspace_settings.name = "sRGB"
        tex.image = img
        tex.extension = 'CLIP'
    except:
        pass

    tco = nt.nodes.new("ShaderNodeTexCoord")

    # ★ [FIX] 변수명 통일 (mapn -> map_node)
    map_node = nt.nodes.new("ShaderNodeMapping")
    map_node.inputs["Scale"].default_value = (1.0 / total_w_m, 1.0 / total_h_m, 1.0)
    map_node.inputs["Location"].default_value = (0.5, 0.5, 0.0)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = 'DOT_PRODUCT'
    dot.inputs[1].default_value = (0.0, 0.0, 1.0)

    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs['From Min'].default_value = -1.0
    mr.inputs['From Max'].default_value = 1.0
    mr.inputs['To Min'].default_value = 0.0
    mr.inputs['To Max'].default_value = 1.0
    mr.clamp = True

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = float(TOP_MASK_LOW)
    ramp.color_ramp.elements[1].position = float(TOP_MASK_HIGH)

    # Links
    nt.links.new(tco.outputs["Object"], map_node.inputs["Vector"])
    nt.links.new(map_node.outputs["Vector"], tex.inputs["Vector"])

    nt.links.new(geom.outputs["Normal"], dot.inputs[0])
    nt.links.new(dot.outputs["Value"], mr.inputs['Value'])
    nt.links.new(mr.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(tex.outputs["Color"], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bs.inputs["Base Color"])
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])

    return m_warp, m_weft, map_node


def setup_camera_and_light(total_w, total_h, n_cols, n_rows):
    scn = bpy.context.scene

    # Camera
    cam_data = bpy.data.cameras.new("WeaveCam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = max(total_w, total_h) * 1.05
    cam = bpy.data.objects.new("WeaveCam", cam_data)
    cam.location = (0, 0, max(total_w, total_h) * 2.0)
    scn.collection.objects.link(cam)
    scn.camera = cam

    # Light
    scale_factor_area = (n_cols * n_rows)
    scale_factor_linear = max(n_cols, n_rows)

    light_data = bpy.data.lights.new("Light_Top", 'AREA')
    light_data.shape = 'RECTANGLE'
    s = max(total_w, total_h)
    light_data.size = max(s * LIGHT_SIZE_MULT, 0.05)
    light_data.size_y = light_data.size
    light_data.energy = LIGHT_TOP_ENERGY_W * scale_factor_area

    light = bpy.data.objects.new("Light_Top", light_data)
    light.location = (0, 0, LIGHT_TOP_HEIGHT_M * scale_factor_linear)
    scn.collection.objects.link(light)


def main():
    set_random_seed(RANDOM_SEED)

    argv = sys.argv
    try:
        idx = argv.index("--")
        tex_path = argv[idx + 1]
        n_cols = int(argv[idx + 2])
        n_rows = int(argv[idx + 3])
        out_path = argv[idx + 4]
    except:
        return

    print(f"Build Start: Grid {n_cols}x{n_rows} ({n_cols * 10}cm x {n_rows * 10}cm)")
    nuke_scene()

    col = bpy.data.collections.new("Weave")
    bpy.context.scene.collection.children.link(col)

    # 전체 물리적 크기 계산 (Grid 개수 * 10cm)
    phys_w_m = n_cols * SWATCH_SIZE_CM * CM2M
    phys_h_m = n_rows * SWATCH_SIZE_CM * CM2M

    # 실 개수 동적 계산
    cnt_warp = int(phys_w_m / warp_pitch_m)
    cnt_weft = int(phys_h_m / weft_pitch_m)

    print(f"   Physical Size: {phys_w_m:.2f}m x {phys_h_m:.2f}m")
    print(f"   Threads: Warp {cnt_warp}ea / Weft {cnt_weft}ea")

    p_warp = make_rect_profile("Profile_Warp", warp_w, warp_t)
    p_weft = make_rect_profile("Profile_Weft", weft_w, weft_t)
    m_warp, m_weft, map_node = setup_materials_reference_style(tex_path, phys_w_m, phys_h_m)

    # Build Warp
    start_x = -phys_w_m / 2 + warp_pitch_m / 2
    for i in range(cnt_warp):
        o = build_thread_optimized(f"Warp_{i}", cnt_weft, 'warp', i, m_warp, p_warp)
        o.location.x = start_x + i * warp_pitch_m
        col.objects.link(o)

    # Build Weft
    start_y = -phys_h_m / 2 + weft_pitch_m / 2
    for j in range(cnt_weft):
        o = build_thread_optimized(f"Weft_{j}", cnt_warp, 'weft', j, m_weft, p_weft, z_offset=WEFT_Z_OFFSET_M)
        o.location.y = start_y + j * weft_pitch_m
        col.objects.link(o)

    empty = bpy.data.objects.new("UV_Target", None)
    bpy.context.scene.collection.objects.link(empty)
    map_node.inputs["Vector"].links[0].from_node.object = empty

    setup_camera_and_light(phys_w_m, phys_h_m, n_cols, n_rows)
    setup_renderer_and_world(out_path, n_cols, n_rows)

    print(f"Render Start -> {out_path}")
    bpy.ops.render.render(write_still=True)
    print("Done.")


if __name__ == "__main__":
    main()