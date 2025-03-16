import bpy
import os
import json

def get_socket_default_value(socket):
    if hasattr(socket, 'default_value'):
        if type(socket.default_value) in [float, int, str]:
            return socket.default_value
        elif hasattr(socket.default_value, 'to_list'):
            return socket.default_value.to_list()  
        else:
            return str(socket.default_value)  
    else:
        return "No default value"

def get_linked_socket_info(socket):
    # Find the link that is connected to this socket
    for link in socket.links:
        # Return the source node and socket name
        return {
            "Source Node": link.from_node.name,
            "Source Socket": link.from_socket.name
        }
    return None

materials_data = []
number_of_materials = len(bpy.data.materials)
print(f"Found {number_of_materials} materials in the project.")

for mat in bpy.data.materials:
    if mat is not None:
        material_info = {
            "Material": mat.name,
            "Node Tree": []
        }

        if mat.node_tree:
            for node in mat.node_tree.nodes:
                node_info = {
                    "Node": node.name,
                    "Type": node.type,
                    "Parameters": {}
                }
                for param in node.inputs:
                    if param.is_linked:
                        linked_info = get_linked_socket_info(param)
                        if linked_info:
                            node_info["Parameters"][param.name] = {
                                "Linked To": linked_info
                            }
                    else:
                        value = get_socket_default_value(param)
                        node_info["Parameters"][param.name] = value
                material_info["Node Tree"].append(node_info)
        else:
            material_info["Node Tree"] = "No node tree in this material."

        materials_data.append(material_info)

blender_file_name = bpy.path.basename(bpy.data.filepath)
base_name = os.path.splitext(blender_file_name)[0]  

output_directory = bpy.path.abspath("//Results\database")  
if not os.path.exists(output_directory):
    os.makedirs(output_directory)  

file_path = os.path.join(output_directory, f"{base_name}_materials_database.json")

try:
    with open(file_path, 'w') as file:
        json.dump(materials_data, file, indent=4)
    print(f"Materials data successfully written to {file_path}")
except IOError as e:
    print(f"Error writing data to file: {e}")
