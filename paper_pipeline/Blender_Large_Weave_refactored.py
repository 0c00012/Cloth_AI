# -*- coding: utf-8 -*-
"""Refactored Blender weave renderer.

Key changes:
- Shared swatch scale is imported from cloth_pipeline_config.py.
- Target physical size is preserved exactly by fitting effective thread pitch.
- Render resolution follows the assembled texture dimensions.
- TexCoord object assignment is handled explicitly and safely.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Optional

import bpy  # type: ignore
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cloth_pipeline_config import RANDOM_SEED, SWATCH_CM

# -----------------------------------------------------------------------------
# Base weave parameters (cm)
# -----------------------------------------------------------------------------
BASE_WARP_PITCH_CM = 0.5
BASE_WARP_WIDTH_CM = 0.35
BASE_WARP_THICK_CM = 0.15

BASE_WEFT_PITCH_CM = 0.8
BASE_WEFT_WIDTH_CM = 0.65
BASE_WEFT_THICK_CM = 0.20
WEFT_Z_OFFSET_M = 0.0003

# Render / lookdev
SAMPLES = 1024
CM_EXPOSURE = 0.05
WORLD_STRENGTH = 0.15
WORLD_COLOR = (1.0, 1.0, 1.0, 1.0)
LIGHT_SIZE_MULT = 1.25
LIGHT_TOP_ENERGY_W = 22.0
LIGHT_TOP_HEIGHT_M = 1.5

# Weave shape
CRIMP_RATIO = 0.32
CRIMP_SCALE = 0.7
STEPS_PER_CELL = 4
BEVEL_RESOLUTION = 3

# Materials
WARP_WHITE = (1.0, 1.0, 1.0, 1.0)
WEFT_SIDE_COLOR = (0.0, 0.0, 0.0, 1.0)
TOP_MASK_LOW = 0.65
TOP_MASK_HIGH = 0.95

CM_TO_M = 0.01


@dataclass(frozen=True)
class WeaveSpec:
    target_width_m: float
    target_height_m: float
    warp_count: int
    weft_count: int
    warp_pitch_m: float
    weft_pitch_m: float
    warp_width_m: float
    warp_thickness_m: float
    weft_width_m: float
    weft_thickness_m: float
    crimp_amp_m: float


def set_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    print(f"Set random seed: {seed}")


def compute_weave_spec(cols: int, rows: int) -> WeaveSpec:
    target_width_m = cols * SWATCH_CM * CM_TO_M
    target_height_m = rows * SWATCH_CM * CM_TO_M

    base_warp_pitch_m = BASE_WARP_PITCH_CM * CM_TO_M
    base_weft_pitch_m = BASE_WEFT_PITCH_CM * CM_TO_M

    warp_count = max(1, round(target_width_m / base_warp_pitch_m))
    weft_count = max(1, round(target_height_m / base_weft_pitch_m))

    warp_pitch_m = target_width_m / warp_count
    weft_pitch_m = target_height_m / weft_count

    avg_pitch = (warp_pitch_m + weft_pitch_m) / 2.0
    crimp_amp_m = avg_pitch * CRIMP_RATIO * CRIMP_SCALE

    return WeaveSpec(
        target_width_m=target_width_m,
        target_height_m=target_height_m,
        warp_count=warp_count,
        weft_count=weft_count,
        warp_pitch_m=warp_pitch_m,
        weft_pitch_m=weft_pitch_m,
        warp_width_m=BASE_WARP_WIDTH_CM * CM_TO_M,
        warp_thickness_m=BASE_WARP_THICK_CM * CM_TO_M,
        weft_width_m=BASE_WEFT_WIDTH_CM * CM_TO_M,
        weft_thickness_m=BASE_WEFT_THICK_CM * CM_TO_M,
        crimp_amp_m=crimp_amp_m,
    )


def nuke_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)

    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.textures,
        bpy.data.lights,
        bpy.data.cameras,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def setup_renderer_and_world(output_path: str, texture_width_px: int, texture_height_px: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SAMPLES
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for device_type in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()
                for device in prefs.devices:
                    device.use = True
                scene.cycles.device = "GPU"
                break
            except Exception:
                continue
        scene.cycles.denoiser = "OPTIX"
        bpy.context.view_layer.cycles.use_denoising = True
    except Exception:
        pass

    view_settings = scene.view_settings
    try:
        view_settings.view_transform = "Filmic"
    except Exception:
        pass
    view_settings.exposure = CM_EXPOSURE
    view_settings.gamma = 1.0

    scene.render.resolution_x = int(texture_width_px)
    scene.render.resolution_y = int(texture_height_px)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.filepath = output_path

    world = bpy.data.worlds.new("WeaveWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    if not bg.outputs["Background"].is_linked:
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = WORLD_COLOR
    bg.inputs["Strength"].default_value = WORLD_STRENGTH


def make_rect_profile(name: str, width_m: float, thickness_m: float) -> bpy.types.Object:
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    spline = curve.splines.new("POLY")
    spline.use_cyclic_u = True

    hw = float(width_m) * 0.5
    ht = float(thickness_m) * 0.5
    points = [(-hw, -ht, 0.0), (hw, -ht, 0.0), (hw, ht, 0.0), (-hw, ht, 0.0)]
    spline.points.add(len(points) - 1)
    for idx, (x, y, z) in enumerate(points):
        spline.points[idx].co = (x, y, z, 1.0)

    return bpy.data.objects.new(name, curve)


def build_thread_curve(
    *,
    name: str,
    crossing_count: int,
    crossing_pitch_m: float,
    axis: str,
    thread_index: int,
    crimp_amp_m: float,
    material: bpy.types.Material,
    profile_object: bpy.types.Object,
    z_offset_m: float = 0.0,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.fill_mode = "FULL"
    spline = curve.splines.new("POLY")

    crossing_count = max(1, crossing_count)
    total_steps = crossing_count * STEPS_PER_CELL
    spline.points.add(total_steps)

    total_len_m = crossing_count * crossing_pitch_m
    half_len = total_len_m / 2.0
    t = np.linspace(0.0, total_len_m, total_steps + 1)

    max_cross_idx = max(crossing_count - 1, 0)
    j = np.minimum((t // crossing_pitch_m).astype(int), max_cross_idx)
    u = np.where(crossing_pitch_m > 0, (t - j * crossing_pitch_m) / crossing_pitch_m, 0.0)

    if axis == "warp":
        x = np.zeros_like(t)
        y = -half_len + t
        phase = (thread_index + j) % 2
        sign = np.where(phase == 0, 1.0, -1.0)
        z = sign * crimp_amp_m * np.sin(pi * u)
    else:
        x = -half_len + t
        y = np.zeros_like(t)
        phase = (j + thread_index) % 2
        sign = np.where(phase == 0, -1.0, 1.0)
        z = sign * crimp_amp_m * np.sin(pi * u) + z_offset_m

    points = np.empty(len(t) * 4, dtype=np.float32)
    points[0::4] = x
    points[1::4] = y
    points[2::4] = z
    points[3::4] = 1.0
    spline.points.foreach_set("co", points)

    obj = bpy.data.objects.new(name, curve)
    obj.data.bevel_mode = "OBJECT"
    obj.data.bevel_object = profile_object
    obj.data.bevel_resolution = BEVEL_RESOLUTION
    obj.data.use_fill_caps = True
    obj.data.materials.append(material)
    return obj


def setup_materials(tex_path: str, total_w_m: float, total_h_m: float):
    # Warp material
    warp_mat = bpy.data.materials.new("Mat_Warp_White")
    warp_mat.use_nodes = True
    nt = warp_mat.node_tree
    nt.nodes.clear()
    output = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = WARP_WHITE
    bsdf.inputs["Roughness"].default_value = 0.8
    nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Weft material with texture on top-facing area
    weft_mat = bpy.data.materials.new("Mat_Weft_Texture")
    weft_mat.use_nodes = True
    nt = weft_mat.node_tree
    nt.nodes.clear()

    output = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    mix = nt.nodes.new("ShaderNodeMixRGB")
    mix.inputs["Color1"].default_value = WEFT_SIDE_COLOR

    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(tex_path)
    img.colorspace_settings.name = "sRGB"
    tex.image = img
    tex.extension = "CLIP"

    tex_coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (
        1.0 / max(total_w_m, 1e-8),
        1.0 / max(total_h_m, 1e-8),
        1.0,
    )
    mapping.inputs["Location"].default_value = (0.5, 0.5, 0.0)

    geom = nt.nodes.new("ShaderNodeNewGeometry")
    dot = nt.nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = (0.0, 0.0, 1.0)

    map_range = nt.nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = -1.0
    map_range.inputs["From Max"].default_value = 1.0
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = float(TOP_MASK_LOW)
    ramp.color_ramp.elements[1].position = float(TOP_MASK_HIGH)

    nt.links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(geom.outputs["Normal"], dot.inputs[0])
    nt.links.new(dot.outputs["Value"], map_range.inputs["Value"])
    nt.links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(tex.outputs["Color"], mix.inputs["Color2"])
    nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    return warp_mat, weft_mat, tex_coord


def setup_camera_and_light(total_w_m: float, total_h_m: float, cols: int, rows: int) -> None:
    scene = bpy.context.scene
    scene.collection.children.link(bpy.data.collections.new("WeaveHelpers"))

    camera_data = bpy.data.cameras.new("WeaveCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(total_w_m, total_h_m) * 1.05
    camera = bpy.data.objects.new("WeaveCamera", camera_data)
    camera.location = (0.0, 0.0, max(total_w_m, total_h_m) * 2.0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new("Light_Top", type="AREA")
    light_data.shape = "RECTANGLE"
    s = max(total_w_m, total_h_m)
    light_data.size = max(s * LIGHT_SIZE_MULT, 0.05)
    light_data.size_y = light_data.size
    light_data.energy = LIGHT_TOP_ENERGY_W * max(cols * rows, 1)

    light = bpy.data.objects.new("Light_Top", light_data)
    light.location = (0.0, 0.0, LIGHT_TOP_HEIGHT_M * max(cols, rows))
    scene.collection.objects.link(light)


def parse_blender_args():
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Missing Blender script arguments after '--'.")
    idx = argv.index("--")
    payload = argv[idx + 1 :]
    if len(payload) < 4:
        raise ValueError(
            "Expected at least 4 arguments: texture_path cols rows out_path [texture_width_px texture_height_px]"
        )

    texture_path = payload[0]
    cols = int(payload[1])
    rows = int(payload[2])
    out_path = payload[3]

    if len(payload) >= 6:
        texture_width_px = int(payload[4])
        texture_height_px = int(payload[5])
    else:
        texture_width_px = cols * 376
        texture_height_px = rows * 376

    return texture_path, cols, rows, out_path, texture_width_px, texture_height_px


def main() -> None:
    set_random_seed(RANDOM_SEED)
    texture_path, cols, rows, out_path, texture_width_px, texture_height_px = parse_blender_args()

    spec = compute_weave_spec(cols, rows)
    print(
        f"Build start: grid {cols}x{rows} | target size {spec.target_width_m:.4f}m x {spec.target_height_m:.4f}m"
    )
    print(
        f"Threads -> warp: {spec.warp_count}, weft: {spec.weft_count} | "
        f"effective pitch -> warp: {spec.warp_pitch_m * 100:.3f}cm, weft: {spec.weft_pitch_m * 100:.3f}cm"
    )

    nuke_scene()
    setup_renderer_and_world(out_path, texture_width_px, texture_height_px)

    weave_collection = bpy.data.collections.new("Weave")
    bpy.context.scene.collection.children.link(weave_collection)

    profile_warp = make_rect_profile("Profile_Warp", spec.warp_width_m, spec.warp_thickness_m)
    profile_weft = make_rect_profile("Profile_Weft", spec.weft_width_m, spec.weft_thickness_m)
    weave_collection.objects.link(profile_warp)
    weave_collection.objects.link(profile_weft)
    profile_warp.hide_render = True
    profile_warp.hide_viewport = True
    profile_weft.hide_render = True
    profile_weft.hide_viewport = True

    mat_warp, mat_weft, tex_coord_node = setup_materials(
        texture_path,
        spec.target_width_m,
        spec.target_height_m,
    )

    start_x = -spec.target_width_m / 2.0 + spec.warp_pitch_m / 2.0
    for idx in range(spec.warp_count):
        obj = build_thread_curve(
            name=f"Warp_{idx}",
            crossing_count=spec.weft_count,
            crossing_pitch_m=spec.weft_pitch_m,
            axis="warp",
            thread_index=idx,
            crimp_amp_m=spec.crimp_amp_m,
            material=mat_warp,
            profile_object=profile_warp,
        )
        obj.location.x = start_x + idx * spec.warp_pitch_m
        weave_collection.objects.link(obj)

    start_y = -spec.target_height_m / 2.0 + spec.weft_pitch_m / 2.0
    for idx in range(spec.weft_count):
        obj = build_thread_curve(
            name=f"Weft_{idx}",
            crossing_count=spec.warp_count,
            crossing_pitch_m=spec.warp_pitch_m,
            axis="weft",
            thread_index=idx,
            crimp_amp_m=spec.crimp_amp_m,
            material=mat_weft,
            profile_object=profile_weft,
            z_offset_m=WEFT_Z_OFFSET_M,
        )
        obj.location.y = start_y + idx * spec.weft_pitch_m
        weave_collection.objects.link(obj)

    uv_target = bpy.data.objects.new("UV_Target", None)
    bpy.context.scene.collection.objects.link(uv_target)
    tex_coord_node.object = uv_target

    setup_camera_and_light(spec.target_width_m, spec.target_height_m, cols, rows)

    print(f"Render start -> {out_path}")
    bpy.ops.render.render(write_still=True)
    print("Render done.")


if __name__ == "__main__":
    main()
