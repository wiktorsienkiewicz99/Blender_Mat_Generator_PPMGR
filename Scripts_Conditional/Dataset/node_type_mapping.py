import bpy

def generate_node_type_map():
    node_type_map = {}
    for node_cls in dir(bpy.types):
        if node_cls.startswith("ShaderNode"):
            try:
                mat = bpy.data.materials.new(name="__temp__")
                mat.use_nodes = True
                node = mat.node_tree.nodes.new(type=node_cls)
                node_type_map[node.type] = node_cls
                bpy.data.materials.remove(mat)
            except:
                continue
    return node_type_map

type_map = generate_node_type_map()

print("NODE_TYPE_MAP = {")
for short, full in sorted(type_map.items()):
    print(f'    "{short}": "{full}",')
print("}")

with open("/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_map.py", "w") as f:
    f.write("NODE_TYPE_MAP = {\n")
    for short, full in sorted(type_map.items()):
        f.write(f'    "{short}": "{full}",\n')
    f.write("}\n")