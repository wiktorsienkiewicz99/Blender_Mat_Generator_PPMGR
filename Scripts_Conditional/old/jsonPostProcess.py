import json

def extract_data_from_json(json_data):
    extracted_data = {}

    for material in json_data:
        material_name = material["Material"]
        material_data = {
            "Nodes": [],
            "Parameters": [],
            "Edges": []
        }
        for node in material["Node Tree"]:
            # Add node details
            material_data["Nodes"].append({
                "Name": node["Node"],
                "Type": node["Type"]
            })

            # Add parameters
            node_parameters = {
                "Node": node["Node"],
                "Parameters": []
            }
            for param, value in node.get("Parameters", {}).items():
                if isinstance(value, dict) and "Linked To" in value:
                    linked_to = value["Linked To"][0]
                    param_value = f"{linked_to['Target Node']}:{linked_to['Target Socket']}"
                else:
                    param_value = value
                node_parameters["Parameters"].append({
                    "Name": param,
                    "Value": param_value
                })
            material_data["Parameters"].append(node_parameters)

            # Add edges
            for output, connections in node.get("Outputs", {}).items():
                if connections != "No connections":
                    for connection in connections:
                        material_data["Edges"].append({
                            "Source": f"{node['Node']}.{output}",
                            "Target": f"{connection['Target Node']}.{connection['Target Socket']}"
                        })

        extracted_data[material_name] = material_data
    return extracted_data

def extract_data_from_json_no_parameters(json_data):
    extracted_data = {}

    for material in json_data:
        material_name = material["Material"]
        material_data = {
            "Nodes": [],
            "Edges": []
        }
        for node in material["Node Tree"]:
            # Add node details
            material_data["Nodes"].append({
                "Name": node["Node"],
                "Type": node["Type"]
            })
            # Add edges
            for output, connections in node.get("Outputs", {}).items():
                if connections != "No connections":
                    for connection in connections:
                        material_data["Edges"].append({
                            "Source": f"{node['Node']}.{output}",
                            "Target": f"{connection['Target Node']}.{connection['Target Socket']}"
                        })

        extracted_data[material_name] = material_data
    return extracted_data

def extract_data_from_json_only_nodes(json_data):
    extracted_data = {}

    for material in json_data:
        material_name = material["Material"]
        material_data ={"Nodes": []}
        for node in material["Node Tree"]:
            # Add node details
            material_data["Nodes"].append({
                "Name": node["Node"],
                "Type": node["Type"]
            })
        extracted_data[material_name] = material_data
    return extracted_data

def read_json_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
def write_json_to_file(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# Path to the JSON file (using raw string to avoid unicode escape errors)
input_file_path = r'C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\database\Mat_Tech_materials_database_test_group.json'
output_file_path = r'C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\database\extracted_materials.json'

input_data = read_json_from_file(input_file_path)


extracted_data = extract_data_from_json_only_nodes(input_data)

write_json_to_file(extracted_data, output_file_path)

'''
for material, data in extracted_data.items():
    print(f"Material: {material}")
    print("Nodes:")
    for node in data["Nodes"]:
        print(f"  - Name: {node['Name']}, Type: {node['Type']}")

    print("\nParameters:")
    for param in data["Parameters"]:
        print(f"  - Node: {param['Node']}")
        for p in param["Parameters"]:
            print(f"    - {p['Name']}: {p['Value']}")

    print("\nEdges:")
    for edge in data["Edges"]:
        print(f"  - Source: {edge['Source']} --> Target: {edge['Target']}")

    print("\n" + "-"*50 + "\n")
'''