import bpy
import os, math, random, itertools
from math import sin, pi
from mathutils import Matrix, Vector

# ========================= (출력/제어) =========================
DO_RENDER     = True
OUTPUT_DIR    = r"C:\Users\_idal\PycharmProjects\Cloth_AI\no2_blender_swatch_maker\render_output\mega_batch_fixed_nolighttouch"
MAX_VARIANTS  = 50          # ▶ 총 생성 이미지 개수 제한
RANDOM_SEED   = 42          # 재현성

# ========================= (원단 이미지 경로) ====================
FABRIC_PATHS = [
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt\KakaoTalk_20230321_093636783_03_crop4.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt\KakaoTalk_20230321_093636783_04_crop3.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt\KakaoTalk_20230321_093803765_03_crop5.png",
    r"C:\Users\_idal\PycharmProjects\Cloth_AI\no1_tshirt_crop\08199_cropped_tshirt\KakaoTalk_20230321_093727482_25_crop3.png",
]

# ========================= (직물 기본 파라미터) ==================
WARP_COUNT = 64
WEFT_COUNT = 64

# 후보 세트(여기 값들로 조합 → 50장에 맞춰 랜덤 샘플)
YARN_SETS = [  # (warp_width, warp_thick, weft_width, weft_thick)
    (0.045, 0.018, 0.055, 0.018),
    (0.055, 0.020, 0.060, 0.020),
    (0.065, 0.025, 0.070, 0.025),
]
PITCH_SETS = [  # (spacing_warp, spacing_weft)
    (0.085, 0.055),
    (0.090, 0.060),
    (0.095, 0.065),
]
CRIMP_SETS = [0.0024, 0.0032, 0.0040]
UV_REPEAT_SETS = [ (1.0,1.0), (2.0,1.0), (1.0,2.0), (2.0,2.0) ]
WEFT_ROTATE_SETS = [False, True]
SHIFT_SETS = [0, 1, 2]

# ========================= (렌더/엔진) ===========================
USE_CYCLES   = True
SAMPLES      = 256
USE_DENOISER = True
CYCLES_DEVICE_TYPE = 'CUDA'

# ========================= (카메라) ==============================
WEAVECAM_LOCATION = (0.0, -4.3352, 7.6820)  # 절대 좌표
LOOK_AT_TARGET    = True
CAMERA_LENS_MM    = 50.0

# ========================= (UV/오프셋) ===========================
UV_PROJECT_FROM_TOP = True
OBJECT_OFFSET       = (0.0, 0.1, 0.0)

# ========================= (고정 머티 색) ========================
WARP_COLOR = (0.08, 0.08, 0.08, 1.0)
WEFT_COLOR = (0.85, 0.85, 0.85, 1.0)

# ================================================================

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def clear_all_objects_except_lights():
    """씬의 모든 오브젝트를 삭제하지만 LIGHT 타입은 절대 건드리지 않음."""
    bpy.ops.object.select_all(action='DESELECT')
    for obj in list(bpy.context.scene.objects):
        if obj.type == 'LIGHT':
            continue
        obj.select_set(True)
    bpy.ops.object.delete()
    # 남아있는 빈 컬렉션 정리(라이트가 속한 컬렉션은 자동 유지됨)
    for col in list(bpy.data.collections):
        if len(col.objects) == 0 and col.users == 0:
            try:
                bpy.data.collections.remove(col)
            except Exception:
                pass

def setup_cycles_engine():
    scene=bpy.context.scene
    if USE_CYCLES:
        scene.render.engine='CYCLES'
        scene.cycles.samples=SAMPLES
        scene.render.film_transparent=True
        if hasattr(scene.cycles,"use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling=True
        if USE_DENOISER:
            try:
                scene.cycles.denoiser='OPTIX'
                bpy.context.view_layer.cycles.use_denoising=True
            except Exception:
                try:
                    scene.cycles.denoiser='OPENIMAGEDENOISE'
                    bpy.context.view_layer.cycles.use_denoising=True
                except Exception: pass
        try:
            prefs=bpy.context.preferences
            cprefs=prefs.addons['cycles'].preferences
            cprefs.use_persistent_data=True
            cprefs.compute_device_type=CYCLES_DEVICE_TYPE
            cprefs.get_devices()
            for d in cprefs.devices: d.use=True
            scene.cycles.device='GPU'
        except Exception as e:
            print("[WARN] GPU 설정 실패:", e)
    else:
        scene.render.engine='BLENDER_EEVEE'

def make_rect_profile(name, width, thickness):
    cu = bpy.data.curves.new(name=name, type='CURVE')
    cu.dimensions='2D'; cu.fill_mode='BOTH'
    sp = cu.splines.new('POLY'); sp.use_cyclic_u=True
    hw,ht = float(width)*0.5, float(thickness)*0.5
    pts=[(-hw,-ht,0),(hw,-ht,0),(hw,ht,0),(-hw,ht,0)]
    sp.points.add(len(pts)-1)
    for i,(x,y,z) in enumerate(pts): sp.points[i].co=(x,y,z,1.0)
    obj=bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(obj)
    return obj

def make_material(name, rgba):
    m=bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes=True; nt=m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out=nt.nodes.new("ShaderNodeOutputMaterial"); out.location=(400,0)
    bsdf=nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location=(150,0)
    bsdf.inputs["Base Color"].default_value=rgba
    bsdf.inputs["Roughness"].default_value=0.8
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return m

def poly_curve(name, points):
    cu=bpy.data.curves.new(name=name, type='CURVE')
    cu.dimensions='3D'; cu.fill_mode='FULL'
    sp=cu.splines.new('POLY')
    sp.points.add(len(points)-1)
    for i,(x,y,z) in enumerate(points): sp.points[i].co=(x,y,z,1.0)
    return bpy.data.objects.new(name, cu)

def build_axis_curve(axis, fixed_pos, count_other, spacing_along, idx_self, mat,
                     width, thickness, steps_per_cell=6, crimp_amp=0.0032, bevel_profile=None):
    half_len=(count_other*spacing_along)*0.5
    total_steps=count_other*steps_per_cell
    pts=[]
    for s in range(total_steps+1):
        t=s/total_steps*(count_other*spacing_along)
        j=min(int(t//spacing_along), count_other-1)
        u=(t - j*spacing_along)/spacing_along
        if axis=='warp':
            x=fixed_pos; y=-half_len+t; over=((idx_self+j)%2)==0; sign=+1 if over else -1
            z=sign*crimp_amp*sin(pi*u)
        else:
            x=-half_len+t; y=fixed_pos; over=((j+idx_self)%2)==0; sign=-1 if over else +1
            z=sign*crimp_amp*sin(pi*u)
        pts.append((x,y,z))
    obj=poly_curve(f"{'Warp' if axis=='warp' else 'Weft'}_{idx_self:02d}", pts)
    c=obj.data; c.bevel_mode='OBJECT'; c.bevel_object=bevel_profile
    c.bevel_resolution=3; c.resolution_u=24
    try: c.twist_mode='Z_UP'
    except Exception: pass
    if c.materials: c.materials[0]=mat
    else: c.materials.append(mat)
    return obj

def auto_clearance_adjust_rect(warp_width, weft_width, spacing_warp, spacing_weft,
                               crimp_amp, warp_thick, weft_thick, margin):
    min_warp_pitch=float(warp_width)*(1.0+margin)
    min_weft_pitch=float(weft_width)*(1.0+margin)
    if spacing_warp<min_warp_pitch: spacing_warp=min_warp_pitch
    if spacing_weft<min_weft_pitch: spacing_weft=min_weft_pitch
    min_center_gap=(float(warp_thick)+float(weft_thick))*(1.0+margin)
    if 2.0*crimp_amp<min_center_gap: crimp_amp=0.5*min_center_gap
    return spacing_warp, spacing_weft, crimp_amp

def ensure_uv(obj, name="UVMap"):
    if obj.type!='MESH': return
    if name not in [l.name for l in obj.data.uv_layers]:
        obj.data.uv_layers.new(name=name)
    obj.data.uv_layers.active=obj.data.uv_layers.get(name)

def uv_project_mod(obj, projector, uv_name="UVMap"):
    if obj.type!='MESH': return
    ensure_uv(obj, uv_name)
    m=obj.modifiers.get("UVProject") or obj.modifiers.new("UVProject", type='UV_PROJECT')
    if len(m.projectors)<1: m.projectors_add()
    m.projectors[0].object=projector
    m.uv_layer=uv_name
    m.aspect_x=m.aspect_y=1.0

def make_image_material_unique(base_name, image_path, repeat=(1,1), rotate_deg=0.0):
    mat=bpy.data.materials.new(f"{base_name}")
    mat.use_nodes=True; nt=mat.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out=nt.nodes.new("ShaderNodeOutputMaterial"); out.location=(1000,0)
    bsdf=nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location=(760,0)
    tex=nt.nodes.new("ShaderNodeTexImage"); tex.location=(540,0)
    mapn=nt.nodes.new("ShaderNodeMapping"); mapn.location=(320,0)
    tco=nt.nodes.new("ShaderNodeTexCoord"); tco.location=(100,0)
    try:
        img=bpy.data.images.load(image_path, check_existing=True)
        tex.image=img; tex.extension='REPEAT'
        try: tex.image.colorspace_settings.name="sRGB"
        except Exception: pass
    except Exception as e:
        print(f"[WARN] 이미지 로드 실패: {image_path} ({e})")
    bsdf.inputs["Roughness"].default_value=1.0
    mapn.inputs["Scale"].default_value[0]=max(1e-6, float(repeat[0]))
    mapn.inputs["Scale"].default_value[1]=max(1e-6, float(repeat[1]))
    mapn.inputs["Rotation"].default_value[2]=math.radians(float(rotate_deg))
    nt.links.new(tco.outputs.get("UV") or tco.outputs["Generated"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def setup_camera_only(total_w, total_h, target):
    """라이트는 절대 건드리지 않고 카메라만 설정."""
    scene=bpy.context.scene
    cam=bpy.data.objects.get("WeaveCam")
    if cam is None:
        cdat=bpy.data.cameras.new("WeaveCam")
        cam=bpy.data.objects.new("WeaveCam", cdat)
        scene.collection.objects.link(cam)
    cam.data.type='PERSP'
    if hasattr(cam.data,"lens"): cam.data.lens=float(CAMERA_LENS_MM)
    cam.location=tuple(map(float, WEAVECAM_LOCATION))
    if LOOK_AT_TARGET:
        forward=(Vector(target)-Vector(cam.location)).normalized()
        world_up=Vector((0,0,1))
        right=forward.cross(world_up).normalized()
        if right.length<1e-6:
            world_up=Vector((0,1,0)); right=forward.cross(world_up).normalized()
        up=right.cross(forward).normalized()
        R=Matrix(((right.x,up.x,-forward.x),
                  (right.y,up.y,-forward.y),
                  (right.z,up.z,-forward.z)))
        cam.rotation_euler=R.to_euler('XYZ')
    scene.camera=cam

def get_uv_projector(total_w, total_h, target, name="UVProjectorTop"):
    smax=max(float(total_w), float(total_h))
    cam=bpy.data.objects.get(name)
    if cam is None:
        c=bpy.data.cameras.new(name); cam=bpy.data.objects.new(name,c)
        bpy.context.scene.collection.objects.link(cam)
    cam.data.type='ORTHO'
    cam.data.ortho_scale=smax*1.5
    cam.location=(float(target[0]), float(target[1]), float(target[2])+1.0)
    cam.rotation_euler=(0,0,0)
    return cam

def generate_plain_weave(warp_count, weft_count, wp, wt, ww, wt2,
                         spacing_warp, spacing_weft, crimp_amp,
                         clearance_margin=0.05, steps_per_cell=6):
    spacing_warp, spacing_weft, crimp_amp = auto_clearance_adjust_rect(
        wp, ww, spacing_warp, spacing_weft, crimp_amp, wt, wt2, clearance_margin)

    col=bpy.data.collections.new("PlainWeave_Swatch")
    bpy.context.scene.collection.children.link(col)

    prof_warp=make_rect_profile("Profile_Warp_Rect", wp, wt)
    prof_weft=make_rect_profile("Profile_Weft_Rect", ww, wt2)
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
        o=build_axis_curve('warp', x, weft_count, spacing_weft, i, mat_warp, wp, wt,
                           steps_per_cell, crimp_amp, prof_warp)
        col.objects.link(o); warp_objs.append(o)
    for j in range(weft_count):
        y=y0 + j*spacing_weft
        o=build_axis_curve('weft', y, warp_count, spacing_warp, j, mat_weft, ww, wt2,
                           steps_per_cell, crimp_amp, prof_weft)
        col.objects.link(o); weft_objs.append(o)

    # 오프셋 적용
    for o in (warp_objs+weft_objs):
        o.location=(o.location.x+OBJECT_OFFSET[0], o.location.y+OBJECT_OFFSET[1], o.location.z+OBJECT_OFFSET[2])

    return dict(total_w=total_w, total_h=total_h, warp_objs=warp_objs, weft_objs=weft_objs)

# -------- 패턴 규칙 --------
def rule_const(k):      return lambda idx, n, sh=0: k % n
def rule_mod():         return lambda idx, n, sh=0: (idx + sh) % n
def rule_lin(a,b):      return lambda i,j,n,sh=0: (a*i + b*j + sh) % n
def rule_checker():     return lambda i,j,n,sh=0: (i + j + sh) % n

def assign_materials_with_rules(warp_objs, weft_objs, proj, warp_mats, weft_mats,
                                n, warp_rule_kind, weft_rule_kind, shift):
    # 곡선 → 메쉬 변환
    bpy.ops.object.select_all(action='DESELECT')
    for o in (warp_objs+weft_objs): o.select_set(True)
    bpy.context.view_layer.objects.active = warp_objs[0]
    bpy.ops.object.convert(target='MESH')

    # UV Project modifier 부착
    for o in warp_objs: uv_project_mod(o, proj)
    for o in weft_objs: uv_project_mod(o, proj)

    # 룰 선택
    if warp_rule_kind == 'const0': w_rule = rule_const(0)
    elif warp_rule_kind == 'const1': w_rule = rule_const(1)
    elif warp_rule_kind == 'mod': w_rule = rule_mod()
    else: w_rule = rule_mod()

    if weft_rule_kind == 'const0': e_rule = rule_const(0)
    elif weft_rule_kind == 'const1': e_rule = rule_const(1)
    elif weft_rule_kind == 'mod': e_rule = rule_mod()
    elif weft_rule_kind == 'checker': e_rule = None
    elif weft_rule_kind == 'lin_i2j': e_rule = None
    else: e_rule = rule_mod()

    nW = len(warp_mats); nE = len(weft_mats)

    # warp
    for i,o in enumerate(warp_objs):
        idx = w_rule(i, n, shift)
        o.data.materials[0] = warp_mats[idx % nW]

    # weft
    for j,o in enumerate(weft_objs):
        if weft_rule_kind == 'checker':
            idx = (j + shift) % n
        elif weft_rule_kind == 'lin_i2j':
            idx = (2*j + shift) % n
        else:
            idx = e_rule(j, n, shift)
        o.data.materials[0] = weft_mats[idx % nE]

def setup_camera_view(total_w, total_h, target):
    setup_camera_only(total_w, total_h, target)
    # 3D뷰를 카메라 시점으로(라이트엔 영향 없음)
    scr=bpy.context.screen
    if scr:
        for area in scr.areas:
            if area.type=='VIEW_3D':
                for space in area.spaces:
                    if space.type=='VIEW_3D':
                        space.region_3d.view_perspective='CAMERA'; break

def run():
    random.seed(RANDOM_SEED)
    ensure_dirs()

    # === 0) 시작 전에: 모든 오브젝트 삭제(라이트는 그대로 유지) ===
    clear_all_objects_except_lights()

    # 엔진 세팅(라이트는 건드리지 않음)
    setup_cycles_engine()

    # 패턴 시나리오(1~4원단)
    scenarios = [
        ("1_one_all_same", 1, 'const0', 'const0'),
        ("2_split_warp0_weft1", 2, 'const0', 'const1'),
        ("2_alt_both", 2, 'mod', 'mod'),
        ("2_alt_warp_only", 2, 'mod', 'const0'),
        ("2_alt_weft_only", 2, 'const0', 'mod'),
        ("3_alt_both", 3, 'mod', 'mod'),
        ("3_warp_mod_weft_const", 3, 'mod', 'const0'),
        ("3_weft_mod_warp_const", 3, 'const0', 'mod'),
        ("4_alt_both", 4, 'mod', 'mod'),
        ("4_checker", 4, 'mod', 'checker'),
        ("4_lin_i_plus_2j", 4, 'mod', 'lin_i2j'),
    ]

    # 모든 조합을 만들되, 섞어서 50개만 실행
    combos = []
    for name, n, wr, er in scenarios:
        if n > len(FABRIC_PATHS): continue
        base = list(range(n))
        perms = [tuple(base)]
        all_perms = list(itertools.permutations(base))
        random.shuffle(all_perms)
        for p in all_perms:
            if p not in perms:
                perms.append(p)
            if len(perms) >= 3:  # 기본 + 2개
                break

        for (wp, wt, ww, wt2) in YARN_SETS:
            for (spw, spv) in PITCH_SETS:
                for crimp in CRIMP_SETS:
                    for uvw in UV_REPEAT_SETS:
                        for uvv in UV_REPEAT_SETS:
                            rot_we = random.choice(WEFT_ROTATE_SETS)
                            shift  = random.choice(SHIFT_SETS)
                            for perm in perms:
                                combos.append((
                                    name, n, wr, er, perm,
                                    wp, wt, ww, wt2,
                                    spw, spv, crimp,
                                    uvw, uvv, rot_we, shift
                                ))

    random.shuffle(combos)
    combos = combos[:MAX_VARIANTS]

    for idx, (name, n, wr, er, perm,
              wp, wt, ww, wt2,
              spw, spv, crimp,
              uvw, uvv, rot_we, shift) in enumerate(combos, 1):

        # ----- 스와치 생성 전, 기존 스와치 관련만 정리(라이트/카메라는 유지) -----
        # PlainWeave_Swatch 컬렉션/프로파일/UVProjector만 제거
        if bpy.data.objects.get("UVProjectorTop"):
            bpy.data.objects.remove(bpy.data.objects["UVProjectorTop"], do_unlink=True)
        col = bpy.data.collections.get("PlainWeave_Swatch")
        if col:
            for o in list(col.objects):
                bpy.data.objects.remove(o, do_unlink=True)
            try:
                bpy.context.scene.collection.children.unlink(col)
            except Exception:
                pass
            try:
                bpy.data.collections.remove(col)
            except Exception:
                pass
        for nm in ["Profile_Warp_Rect", "Profile_Weft_Rect"]:
            if bpy.data.objects.get(nm):
                bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)

        # ----- 스와치 생성 -----
        result = generate_plain_weave(
            WARP_COUNT, WEFT_COUNT,
            wp, wt, ww, wt2,
            spw, spv, crimp,
            clearance_margin=0.05, steps_per_cell=6
        )
        target = OBJECT_OFFSET
        setup_camera_view(result['total_w'], result['total_h'], target)
        projector = get_uv_projector(result['total_w'], result['total_h'], target) if UV_PROJECT_FROM_TOP else bpy.data.objects.get("WeaveCam")

        # ----- 머티리얼(이 베리에이션 전용) -----
        selected_paths = [FABRIC_PATHS[i] for i in perm]
        warp_mats = []
        weft_mats = []
        for k, p in enumerate(selected_paths):
            mw = make_image_material_unique(f"Var{idx:03d}_Warp_{k}", p, repeat=uvw, rotate_deg=0.0)
            me = make_image_material_unique(f"Var{idx:03d}_Weft_{k}", p, repeat=uvv, rotate_deg=(90.0 if rot_we else 0.0))
            warp_mats.append(mw)
            weft_mats.append(me)

        # ----- 패턴 적용 & 렌더 -----
        assign_materials_with_rules(
            result['warp_objs'], result['weft_objs'], projector,
            warp_mats, weft_mats, n,
            warp_rule_kind=wr, weft_rule_kind=er, shift=shift
        )

        if DO_RENDER:
            tag = (
                f"{idx:03d}_{name}_n{n}"
                f"_perm{''.join(map(str,perm))}"
                f"_wp{wp:.3f}_wt{wt:.3f}_ww{ww:.3f}_wt2{wt2:.3f}"
                f"_spw{spw:.3f}_spv{spv:.3f}"
                f"_cr{crimp:.4f}"
                f"_uvw{uvw[0]}x{uvw[1]}_uvv{uvv[0]}x{uvv[1]}"
                f"_weRot{'90' if rot_we else '0'}_sh{shift}"
            ).replace('.', 'p')
            out_path = os.path.join(OUTPUT_DIR, f"swatch_{tag}.png")
            bpy.context.scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)
            print(f"[{idx:02d}/{len(combos)}] Saved:", out_path)
        else:
            print(f"[{idx:02d}/{len(combos)}] Applied (미렌더)")

if __name__ == "__main__":
    run()
