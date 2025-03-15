import bpy
import mathutils

# Remove stuff
try:
    cube = bpy.data.objects['Cube']
    bpy.data.objects.remove(cube, do_unlink=True)
except:
    print("Object bpy.data.objects['Cube'] not found")

bpy.ops.outliner.orphans_purge()


# Declare constructors
def new_sphere(mylocation, myradius, myname):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64,
        ring_count=32,
        radius=myradius,
        location=mylocation)
    current_name = bpy.context.selected_objects[0].name
    sphere = bpy.data.objects[current_name]
    sphere.name = myname
    sphere.data.name = myname + "_mesh"
    return


def new_plane(mylocation, mysize, myname):
    bpy.ops.mesh.primitive_plane_add(
        size=mysize,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=mylocation,
        rotation=(0, 0, 0),
        scale=(0, 0, 0))
    current_name = bpy.context.selected_objects[0].name
    plane = bpy.data.objects[current_name]
    plane.name = myname
    plane.data.name = myname + "_mesh"
    return


# Create objects
new_sphere((0, 0, 0), 1.0, "MySphere")
new_plane((0, 0, -1), 10, "MyFloor")
sphere = bpy.data.objects['MySphere']
plane = bpy.data.objects['MyFloor']
# Smoothen sphere
for poly in sphere.data.polygons:
    poly.use_smooth = True
# Create TackyPlastic material
MAT_NAME = "TackyPlastic"
bpy.data.materials.new(MAT_NAME)
material = bpy.data.materials[MAT_NAME]
material.use_nodes = True
material.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.2
material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (1, 0, 1, 1)
# Associate TackyPlastic to sphere
if len(sphere.data.materials.items()) != 0:
    sphere.data.materials.clear()
else:
    sphere.data.materials.append(material)
# Create TackyGold material
MAT_NAME = "TackyGold"
bpy.data.materials.new(MAT_NAME)
material = bpy.data.materials[MAT_NAME]
material.use_nodes = True
material.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.1
material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.75, 0.5, 0.05, 1)
material.node_tree.nodes['Principled BSDF'].inputs['Metallic'].default_value = 0.9
# Associate TackyGold to plane
if len(plane.data.materials.items()) != 0:
    plane.data.materials.clear()
else:
    plane.data.materials.append(material)
# Lighten the world light
bpy.data.worlds['World'].node_tree.nodes["Background"].inputs[0].default_value = (0.7, 0.7, 0.7, 1)
# Move camera
cam = bpy.data.objects['Camera']
cam.location = cam.location + mathutils.Vector((0.1, 0, 0))

bpy.context.scene.render.filepath = "./test.png"
bpy.ops.render.render(use_viewport=True, write_still=True)