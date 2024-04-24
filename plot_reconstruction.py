import bpy
import blender_plots as bplt
import blender_plots.blender_utils as bu
import numpy as np
from tqdm import tqdm

def srgb_to_linearrgb(c):
    if   c < 0:       return 0
    elif c < 0.04045: return c/12.92
    else:             return ((c+0.055)/1.055)**2.4


def hex_to_rgb(h,alpha=1):
    # source: https://blender.stackexchange.com/questions/153094/blender-2-8-python-how-to-set-material-color-using-hex-value-instead-of-rgb
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return tuple([srgb_to_linearrgb(c/0xff) for c in (r,g,b)] + [alpha])

orange1 = hex_to_rgb(0xff9a00)
orange2 = hex_to_rgb(0xff5d00)
blue1 = hex_to_rgb(0x00a2ff)
blue2 = hex_to_rgb(0x0065ff)
white = [1, 1, 1]


def get_frustum(intrinsics, height, width, image_depth, name="", with_fill=True, thickness=0.03):
    frustum_points = np.array([
        [0, height, 1],
        [width, height, 1],
        [0, 0, 1],
        [width, 0, 1]
    ]) * image_depth

    frustum_points = np.einsum('ij,...j->...i',
        np.linalg.inv(intrinsics),
        frustum_points,
    )
    frustum_edges = np.array([
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],
        [0, 4],
        [1, 4],
        [2, 4],
        [3, 4]
    ])

    frustum_faces = [
        [0, 1, 4],
        [1, 3, 4],
        [3, 2, 4],
        [2, 0, 4],
    ]

    mesh = bpy.data.meshes.new("frustum")
    mesh.from_pydata(np.vstack([frustum_points, np.zeros(3)]), frustum_edges, frustum_faces)
    frustum = bu.new_empty("frustum" + name, mesh)
    modifier = bu.add_modifier(frustum, "WIREFRAME", use_crease=True, crease_weight=0.6, thickness=thickness, use_boundary=True)
    bpy.context.view_layer.objects.active = frustum
    bpy.ops.object.modifier_apply(modifier=modifier.name)

    if with_fill:
        mesh_fill = bpy.data.meshes.new("fill")
        mesh_fill.from_pydata(np.vstack([frustum_points, np.zeros(3)]), frustum_edges, frustum_faces + [[0, 1, 3, 2]])
        fill = bu.new_empty("fill" + name, mesh_fill)
    else:
        fill = None
    return fill, frustum

def set_color(mesh, color):
    if len(color) == 3:
        color = [*color, 1.]
    mesh.materials.append(bpy.data.materials.new("color"))
    mesh.materials[0].use_nodes = True
    mesh.materials[0].node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = color

def plot_cameras(R, t, intrinsics, height, width, image_depth, name="", with_fill=True, **kwargs):
    fill, frustum = get_frustum(intrinsics, height, width, image_depth, name, with_fill, **kwargs)
    if with_fill:
        s = bplt.Scatter(
            t,
            marker_rotation=R,
            marker_type=fill,
            color=orange2,
            name="fill_scatter" + name,
        )
    s = bplt.Scatter(
        t,
        marker_rotation=R,
        marker_type=frustum,
        name="frustum_scatter" + name,
        color=orange1,
    )

def setup_scene(floor_z=None, resolution=None, sun_energy=1):
    if floor_z is not None:
        floor_size = 500
        floor = bplt.Scatter([0, 0, floor_z], marker_type='cubes', size=(floor_size, floor_size, 0.1), name='floor')
        floor.base_object.is_shadow_catcher = True

    if resolution == 'thin':
        resolution = (1363, 2592)
    if resolution is not None:
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]

    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.data.scenes["Scene"].cycles.samples = 256


    if "Sun" in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects["Sun"])
    bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.data.objects["Sun"].data.energy = sun_energy
    bpy.data.objects["Sun"].data.angle = 0
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0