# -*- coding: utf-8 -*-
"""
Close-up render of the yarn crimp geometry (figure support for paper 3.6.3).

Builds ONE 10 x 10 cm cell of the woven structure with exactly the same
functions, yarn dimensions, crimp parameters and materials as
Blender_Large_Weave.py, then renders it with a perspective camera from
(a) an oblique view and (b) a low grazing view so the alternating over-under
crimp of the warp and weft yarns is visible. No UI, no geometry changes.

Run (from Final_pipeline):
  blender -b -P render_crimp_closeup.py -- <texture.png> <output_dir> [samples]

The texture should be a single 376 x 376 px swatch (one 10 x 10 cm cell), so the
pattern scale matches the physical scale of the paper pipeline.
"""
import importlib.util
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("weave", os.path.join(HERE, "Blender_Large_Weave.py"))
weave = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weave)


def look_at(obj, target):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def build_patch(tex_path, n_cols=1, n_rows=1):
    weave.nuke_scene()
    col = bpy.data.collections.new("Weave")
    bpy.context.scene.collection.children.link(col)
    phys_w = n_cols * weave.SWATCH_SIZE_CM * weave.CM2M
    phys_h = n_rows * weave.SWATCH_SIZE_CM * weave.CM2M
    cnt_warp = int(phys_w / weave.warp_pitch_m)
    cnt_weft = int(phys_h / weave.weft_pitch_m)
    p_warp = weave.make_rect_profile("Profile_Warp", weave.warp_w, weave.warp_t)
    p_weft = weave.make_rect_profile("Profile_Weft", weave.weft_w, weave.weft_t)
    m_warp, m_weft, map_node = weave.setup_materials_reference_style(tex_path, phys_w, phys_h)
    start_x = -phys_w / 2 + weave.warp_pitch_m / 2
    for i in range(cnt_warp):
        o = weave.build_thread_optimized(f"Warp_{i}", cnt_weft, "warp", i, m_warp, p_warp)
        o.location.x = start_x + i * weave.warp_pitch_m
        col.objects.link(o)
    start_y = -phys_h / 2 + weave.weft_pitch_m / 2
    for j in range(cnt_weft):
        o = weave.build_thread_optimized(f"Weft_{j}", cnt_warp, "weft", j, m_weft, p_weft, z_offset=weave.WEFT_Z_OFFSET_M)
        o.location.y = start_y + j * weave.weft_pitch_m
        col.objects.link(o)
    empty = bpy.data.objects.new("UV_Target", None)
    bpy.context.scene.collection.objects.link(empty)
    map_node.inputs["Vector"].links[0].from_node.object = empty
    print(f"patch {phys_w*100:.0f} x {phys_h*100:.0f} cm: warp {cnt_warp}, weft {cnt_weft}")
    return phys_w, phys_h


def setup_world_and_render(samples, out_path, res=(2400, 1600)):
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.samples = samples
    scn.cycles.use_adaptive_sampling = True
    scn.cycles.adaptive_threshold = 0.01
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev_type in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dev_type
                prefs.get_devices()
                for d in prefs.devices:
                    d.use = True
                scn.cycles.device = "GPU"
                break
            except Exception:
                pass
        scn.cycles.denoiser = "OPTIX"
        bpy.context.view_layer.cycles.use_denoising = True
    except Exception:
        pass
    vs = scn.view_settings
    try:
        vs.view_transform = "Filmic"
    except Exception:
        pass
    vs.exposure = weave.CM_EXPOSURE
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGBA"
    scn.render.film_transparent = True
    scn.render.filepath = out_path
    world = bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = weave.WORLD_COLOR
    bg.inputs["Strength"].default_value = weave.WORLD_STRENGTH


def add_camera_and_light(phys_w, phys_h, elevation_deg, azimuth_deg, distance_mult, focal_mm=50):
    scn = bpy.context.scene
    cam_data = bpy.data.cameras.new("CloseupCam")
    cam_data.type = "PERSP"
    cam_data.lens = focal_mm
    cam = bpy.data.objects.new("CloseupCam", cam_data)
    r = max(phys_w, phys_h) * distance_mult
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    cam.location = (r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el))
    scn.collection.objects.link(cam)
    look_at(cam, (0, 0, 0))
    scn.camera = cam
    light_data = bpy.data.lights.new("Light_Top", "AREA")
    light_data.shape = "RECTANGLE"
    s = max(phys_w, phys_h)
    light_data.size = light_data.size_y = max(s * weave.LIGHT_SIZE_MULT, 0.05)
    light_data.energy = weave.LIGHT_TOP_ENERGY_W
    light = bpy.data.objects.new("Light_Top", light_data)
    # same rule as Blender_Large_Weave.setup_camera_and_light for a 1 x 1 grid: 22 W at 1.5 m, pointing down
    light.location = (0, 0, weave.LIGHT_TOP_HEIGHT_M)
    scn.collection.objects.link(light)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    tex_path, out_dir = os.path.abspath(argv[0]), os.path.abspath(argv[1])   # Blender resolves relative paths oddly
    samples = int(argv[2]) if len(argv) > 2 else 512
    os.makedirs(out_dir, exist_ok=True)
    weave.set_random_seed(weave.RANDOM_SEED)
    views = [("crimp_closeup_a_oblique.png", 38, -60, 1.35, 50),
             ("crimp_closeup_b_grazing.png", 12, -75, 1.15, 70)]
    for name, elev, az, dist, focal in views:
        phys_w, phys_h = build_patch(tex_path)
        add_camera_and_light(phys_w, phys_h, elev, az, dist, focal)
        setup_world_and_render(samples, os.path.join(out_dir, name))
        print("render ->", name)
        bpy.ops.render.render(write_still=True)
    print("Done.")


if __name__ == "__main__":
    main()
