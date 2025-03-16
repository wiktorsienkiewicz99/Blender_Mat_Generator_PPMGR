import bpy

def create_material_from_file(file_path):
    material = bpy.data.materials.new(name="Imported Material")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
                #magic goes here
            pass
    return material

def create_icosphere_with_material(material):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1)
    icosphere = bpy.context.active_object
    icosphere.data.materials.append(material)

# Usage
file_path = "/path/to/your/material_export.txt"
material = create_material_from_file(file_path)
create_icosphere_with_material(material)