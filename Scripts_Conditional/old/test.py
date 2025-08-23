import bpy
import os

# Clear the console in Windows
os.system('cls' if os.name == 'nt' else 'clear')

# Get the current scene
scene = bpy.context.scene

print("\nMaterials Information in the Scene:")

# Function to safely get the default value of a node socket
def get_socket_default_value(socket):
    if hasattr(socket, 'default_value'):
        return socket.default_value
    else:
        return "Not Available"

# Ensure the directory for saving the files exists and is writable
output_directory = bpy.path.abspath("//")  # This sets the directory to the current blend file's location
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over all objects in the scene
for obj in scene.objects:
    print("\nObject: ", obj.name)  # Print the name of the object

    # Check if the object has materials
    if obj.data.materials:
        # Iterate over all materials of the object
        for mat in obj.data.materials:
            if mat is not None:
                material_info = "Material: " + mat.name + "\n"
                file_name = f"{mat.name.replace(' ', '_')}_export.txt"
                file_path = os.path.join(output_directory, file_name)

                # Check if the material has a node tree
                if mat.node_tree:
                    material_info += "  Node Tree:\n"
                    for node in mat.node_tree.nodes:
                        node_info = f"    Node: {node.name} ({node.type})\n"
                        # Get parameters of the node
                        for param in node.inputs:
                            if param.is_linked:
                                node_info += f"      {param.name}: Linked\n"
                            else:
                                value = get_socket_default_value(param)
                                node_info += f"      {param.name}: {value}\n"
                        material_info += node_info
                else:
                    material_info += "  No node tree in this material.\n"

                # Write material information to a file
                try:
                    with open(file_path, 'w') as file:
                        file.write(material_info)
                except IOError as e:
                    print(f"Error writing file {file_name}: {e}")

                print(material_info)
            else:
                print("  Material: None assigned")
    else:
        print("  No materials assigned.")

print("\nEnd of material information.")
