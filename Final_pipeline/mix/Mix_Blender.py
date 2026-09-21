from __future__ import annotations

import sys
from dataclasses import dataclass
from math import pi
from pathlib import Path

import bpy
import numpy as np


CM2M = 0.01


@dataclass(frozen=True)
class WeaveSettings:
    swatch_size_cm: float = 10.0
    random_seed: int | None = 42

    cell_pitch_cm: float = 1.0
    weft_width_ratio: float = 0.65
    warp_width_scale: float = 0.75
    width_tighten_factor: float = 1.15
    max_width_ratio_cap: float = 0.95
    thick_ratio: float = 0.25
    weft_z_offset_m: float = 0.0003

    samples: int = 1024
    exposure: float = 0.05
    resolution_scale: float = 1.0

    world_strength: float = 0.15
    world_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    light_size_mult: float = 1.25
    light_top_energy_w: float = 22.0
    light_top_height_m: float = 1.5

    crimp_ratio: float = 0.32
    crimp_scale: float = 0.7

    steps_per_cell: int = 4
    bevel_resolution: int = 3

    weft_side_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    top_mask_low: float = 0.65
    top_mask_high: float = 0.95


@dataclass(frozen=True)
class GeometryConfig:
    warp_pitch_m: float
    weft_pitch_m: float
    warp_width_m: float
    weft_width_m: float
    warp_thickness_m: float
    weft_thickness_m: float
    crimp_amplitude_m: float


@dataclass(frozen=True)
class RenderJob:
    warp_texture_path: Path
    weft_texture_path: Path
    cols: int
    rows: int
    output_path: Path

    def validate(self) -> "RenderJob":
        if not self.warp_texture_path.exists():
            raise FileNotFoundError(f"Warp texture does not exist: {self.warp_texture_path}")
        if not self.weft_texture_path.exists():
            raise FileNotFoundError(f"Weft texture does not exist: {self.weft_texture_path}")
        if self.cols <= 0 or self.rows <= 0:
            raise ValueError(f"Grid size must be positive, got {self.cols}x{self.rows}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def physical_width_m(self) -> float:
        return self.cols * SETTINGS.swatch_size_cm * CM2M

    @property
    def physical_height_m(self) -> float:
        return self.rows * SETTINGS.swatch_size_cm * CM2M


@dataclass
class MaterialBundle:
    warp_material: object
    weft_material: object
    warp_tex_coord: object
    weft_tex_coord: object


SETTINGS = WeaveSettings()


def derive_geometry(settings: WeaveSettings) -> GeometryConfig:
    pitch_m = settings.cell_pitch_cm * CM2M
    weft_width_m = pitch_m * min(
        settings.max_width_ratio_cap,
        settings.weft_width_ratio * settings.width_tighten_factor,
    )
    return GeometryConfig(
        warp_pitch_m=pitch_m,
        weft_pitch_m=pitch_m,
        warp_width_m=weft_width_m * settings.warp_width_scale,
        weft_width_m=weft_width_m,
        warp_thickness_m=pitch_m * settings.thick_ratio,
        weft_thickness_m=pitch_m * settings.thick_ratio,
        crimp_amplitude_m=pitch_m * settings.crimp_ratio * settings.crimp_scale,
    )


GEOMETRY = derive_geometry(SETTINGS)


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return

    import random

    random.seed(seed)
    np.random.seed(seed)
    print(f"Set random seed: {seed}")


def parse_args(argv: list[str]) -> RenderJob:
    try:
        separator_index = argv.index("--")
        warp_texture_path = Path(argv[separator_index + 1]).resolve()
        weft_texture_path = Path(argv[separator_index + 2]).resolve()
        cols = int(argv[separator_index + 3])
        rows = int(argv[separator_index + 4])
        output_path = Path(argv[separator_index + 5]).resolve()
    except (ValueError, IndexError) as exc:
        raise ValueError(
            "Usage: blender -b -P Mix_Blender.py -- "
            "<warp_texture> <weft_texture> <cols> <rows> <output_path>"
        ) from exc

    return RenderJob(
        warp_texture_path=warp_texture_path,
        weft_texture_path=weft_texture_path,
        cols=cols,
        rows=rows,
        output_path=output_path,
    ).validate()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    data_groups = (
        bpy.data.materials,
        bpy.data.images,
        bpy.data.textures,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.worlds,
    )
    for data_group in data_groups:
        for block in list(data_group):
            data_group.remove(block, do_unlink=True)


def enable_gpu_if_available(scene: bpy.types.Scene) -> None:
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return

    for device_type in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = device_type
            preferences.get_devices()
            for device in preferences.devices:
                device.use = True

            if preferences.devices:
                scene.cycles.device = "GPU"
                try:
                    scene.cycles.denoiser = "OPTIX"
                    bpy.context.view_layer.cycles.use_denoising = True
                except Exception:
                    pass
                print(f"Cycles device: {device_type}")
                return
        except Exception as exc:
            print(f"Cycles device setup skipped for {device_type}: {exc}")


def setup_renderer_and_world(output_path: Path, cols: int, rows: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = SETTINGS.samples

    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01

    enable_gpu_if_available(scene)

    view_settings = scene.view_settings
    try:
        view_settings.view_transform = "Filmic"
    except Exception:
        pass
    view_settings.exposure = SETTINGS.exposure

    base_px = 376
    scene.render.resolution_x = int(base_px * cols * SETTINGS.resolution_scale)
    scene.render.resolution_y = int(base_px * rows * SETTINGS.resolution_scale)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.filepath = str(output_path)

    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    node_tree = world.node_tree

    background = node_tree.nodes.get("Background")
    if background is None:
        background = node_tree.nodes.new("ShaderNodeBackground")

    world_output = node_tree.nodes.get("World Output")
    if world_output is None:
        world_output = node_tree.nodes.new("ShaderNodeOutputWorld")
    node_tree.links.new(background.outputs["Background"], world_output.inputs["Surface"])

    background.inputs["Color"].default_value = SETTINGS.world_color
    background.inputs["Strength"].default_value = SETTINGS.world_strength


def make_rect_profile(name: str, width: float, thickness: float) -> bpy.types.Object:
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "2D"
    curve.fill_mode = "BOTH"
    spline = curve.splines.new("POLY")
    spline.use_cyclic_u = True

    half_width = width * 0.5
    half_thickness = thickness * 0.5
    points = [
        (-half_width, -half_thickness, 0.0),
        (half_width, -half_thickness, 0.0),
        (half_width, half_thickness, 0.0),
        (-half_width, half_thickness, 0.0),
    ]
    spline.points.add(len(points) - 1)
    for index, (x_value, y_value, z_value) in enumerate(points):
        spline.points[index].co = (x_value, y_value, z_value, 1.0)

    return bpy.data.objects.new(name, curve)


def build_thread_curve(
    name: str,
    crossing_count: int,
    axis: str,
    index: int,
    material: bpy.types.Material,
    profile: bpy.types.Object,
    z_offset: float = 0.0,
) -> bpy.types.Object:
    crossing_pitch = GEOMETRY.weft_pitch_m if axis == "warp" else GEOMETRY.warp_pitch_m

    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.fill_mode = "FULL"
    spline = curve.splines.new("POLY")

    total_steps = max(1, crossing_count * SETTINGS.steps_per_cell)
    spline.points.add(total_steps)

    total_length = crossing_count * crossing_pitch
    half_length = total_length / 2.0

    t_values = np.linspace(0.0, total_length, total_steps + 1)
    crossing_indices = np.minimum((t_values // crossing_pitch).astype(int), crossing_count - 1)
    normalized = (t_values - crossing_indices * crossing_pitch) / crossing_pitch

    if axis == "warp":
        x_values = np.zeros_like(t_values)
        y_values = -half_length + t_values
        phase = (index + crossing_indices) % 2
        direction = np.where(phase == 0, 1.0, -1.0)
        z_values = direction * GEOMETRY.crimp_amplitude_m * np.sin(pi * normalized)
    else:
        x_values = -half_length + t_values
        y_values = np.zeros_like(t_values)
        phase = (crossing_indices + index) % 2
        direction = np.where(phase == 0, -1.0, 1.0)
        z_values = direction * GEOMETRY.crimp_amplitude_m * np.sin(pi * normalized) + z_offset

    packed_points = np.empty(len(t_values) * 4, dtype=np.float32)
    packed_points[0::4] = x_values
    packed_points[1::4] = y_values
    packed_points[2::4] = z_values
    packed_points[3::4] = 1.0
    spline.points.foreach_set("co", packed_points)

    obj = bpy.data.objects.new(name, curve)
    obj.data.bevel_mode = "OBJECT"
    obj.data.bevel_object = profile
    obj.data.bevel_resolution = SETTINGS.bevel_resolution
    obj.data.use_fill_caps = True
    obj.data.materials.append(material)
    return obj


def create_textured_material(
    name: str,
    texture_path: Path,
    total_width_m: float,
    total_height_m: float,
) -> tuple[bpy.types.Material, bpy.types.ShaderNodeTexCoord]:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    node_tree = material.node_tree
    node_tree.nodes.clear()

    output = node_tree.nodes.new("ShaderNodeOutputMaterial")
    principled = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 0.8

    mix_node = node_tree.nodes.new("ShaderNodeMixRGB")
    mix_node.inputs["Color1"].default_value = SETTINGS.weft_side_color

    image_node = node_tree.nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.load(str(texture_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image_node.image = image
    image_node.extension = "CLIP"

    tex_coord = node_tree.nodes.new("ShaderNodeTexCoord")
    mapping = node_tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (
        1.0 / total_width_m,
        1.0 / total_height_m,
        1.0,
    )
    mapping.inputs["Location"].default_value = (0.5, 0.5, 0.0)

    geometry = node_tree.nodes.new("ShaderNodeNewGeometry")
    dot_product = node_tree.nodes.new("ShaderNodeVectorMath")
    dot_product.operation = "DOT_PRODUCT"
    dot_product.inputs[1].default_value = (0.0, 0.0, 1.0)

    map_range = node_tree.nodes.new("ShaderNodeMapRange")
    map_range.inputs["From Min"].default_value = -1.0
    map_range.inputs["From Max"].default_value = 1.0
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    map_range.clamp = True

    color_ramp = node_tree.nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = SETTINGS.top_mask_low
    color_ramp.color_ramp.elements[1].position = SETTINGS.top_mask_high

    node_tree.links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    node_tree.links.new(mapping.outputs["Vector"], image_node.inputs["Vector"])
    node_tree.links.new(geometry.outputs["Normal"], dot_product.inputs[0])
    node_tree.links.new(dot_product.outputs["Value"], map_range.inputs["Value"])
    node_tree.links.new(map_range.outputs["Result"], color_ramp.inputs["Fac"])
    node_tree.links.new(color_ramp.outputs["Color"], mix_node.inputs["Fac"])
    node_tree.links.new(image_node.outputs["Color"], mix_node.inputs["Color2"])
    node_tree.links.new(mix_node.outputs["Color"], principled.inputs["Base Color"])
    node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    return material, tex_coord


def setup_materials(job: RenderJob) -> MaterialBundle:
    warp_material, warp_tex_coord = create_textured_material(
        "Mat_Warp",
        job.warp_texture_path,
        job.physical_width_m,
        job.physical_height_m,
    )
    weft_material, weft_tex_coord = create_textured_material(
        "Mat_Weft",
        job.weft_texture_path,
        job.physical_width_m,
        job.physical_height_m,
    )
    return MaterialBundle(
        warp_material=warp_material,
        weft_material=weft_material,
        warp_tex_coord=warp_tex_coord,
        weft_tex_coord=weft_tex_coord,
    )


def setup_camera_and_light(total_width_m: float, total_height_m: float, cols: int, rows: int) -> None:
    scene = bpy.context.scene
    size = max(total_width_m, total_height_m)

    camera_data = bpy.data.cameras.new("WeaveCam")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = size * 1.05
    camera = bpy.data.objects.new("WeaveCam", camera_data)
    camera.location = (0.0, 0.0, size * 2.0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    light_data = bpy.data.lights.new("Light_Top", "AREA")
    light_data.shape = "RECTANGLE"
    light_data.size = max(size * SETTINGS.light_size_mult, 0.05)
    light_data.size_y = light_data.size
    light_data.energy = SETTINGS.light_top_energy_w * (cols * rows)

    light = bpy.data.objects.new("Light_Top", light_data)
    light.location = (0.0, 0.0, SETTINGS.light_top_height_m * max(cols, rows))
    scene.collection.objects.link(light)


def attach_uv_target(materials: MaterialBundle) -> None:
    target = bpy.data.objects.new("UV_Target", None)
    bpy.context.scene.collection.objects.link(target)
    materials.warp_tex_coord.object = target
    materials.weft_tex_coord.object = target


def main() -> int:
    try:
        job = parse_args(sys.argv)
    except Exception as exc:
        print(f"[Error] {exc}")
        return 1

    set_random_seed(SETTINGS.random_seed)
    clear_scene()

    weave_collection = bpy.data.collections.new("Weave")
    bpy.context.scene.collection.children.link(weave_collection)

    warp_count = max(1, int(job.physical_width_m / GEOMETRY.warp_pitch_m))
    weft_count = max(1, int(job.physical_height_m / GEOMETRY.weft_pitch_m))

    print(f"Build start: Grid {job.cols}x{job.rows}")
    print(f"Physical size: {job.physical_width_m:.2f}m x {job.physical_height_m:.2f}m")
    print(f"Threads: Warp {warp_count} / Weft {weft_count}")

    warp_profile = make_rect_profile(
        "Profile_Warp",
        GEOMETRY.warp_width_m,
        GEOMETRY.warp_thickness_m,
    )
    weft_profile = make_rect_profile(
        "Profile_Weft",
        GEOMETRY.weft_width_m,
        GEOMETRY.weft_thickness_m,
    )
    materials = setup_materials(job)

    start_x = -job.physical_width_m / 2.0 + GEOMETRY.warp_pitch_m / 2.0
    for index in range(warp_count):
        warp_thread = build_thread_curve(
            f"Warp_{index}",
            weft_count,
            "warp",
            index,
            materials.warp_material,
            warp_profile,
        )
        warp_thread.location.x = start_x + index * GEOMETRY.warp_pitch_m
        weave_collection.objects.link(warp_thread)

    start_y = -job.physical_height_m / 2.0 + GEOMETRY.weft_pitch_m / 2.0
    for index in range(weft_count):
        weft_thread = build_thread_curve(
            f"Weft_{index}",
            warp_count,
            "weft",
            index,
            materials.weft_material,
            weft_profile,
            z_offset=SETTINGS.weft_z_offset_m,
        )
        weft_thread.location.y = start_y + index * GEOMETRY.weft_pitch_m
        weave_collection.objects.link(weft_thread)

    attach_uv_target(materials)
    setup_camera_and_light(job.physical_width_m, job.physical_height_m, job.cols, job.rows)
    setup_renderer_and_world(job.output_path, job.cols, job.rows)

    print(f"Render start -> {job.output_path}")
    bpy.ops.render.render(write_still=True)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
