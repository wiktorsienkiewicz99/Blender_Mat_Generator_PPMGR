import json

# Paths
edge_database_path = r"C:\Users\hyperbook\Desktop\PPMGR\cleaned_edge_database.json"
preprocessed_data_path = r"C:\Users\hyperbook\Desktop\PPMGR\preprocessed_edge_data.json"
node_to_id_path = r"C:\Users\hyperbook\Desktop\PPMGR\node_to_id.json"

def preprocess_edge_database(input_file, output_file, node_to_id_file):
    """Preprocess the edge database to replace node types with IDs and format edges."""
    with open(input_file, "r") as file:
        data = json.load(file)

    with open(node_to_id_file, "r") as file:
        node_to_id = json.load(file)

    preprocessed_data = []

    for material_name, material_data in data.get("materials", {}).items():
        edges = material_data.get("edges", [])

        if not isinstance(edges, list):
            print(f"Invalid edges format in material '{material_name}': {type(edges)}")
            continue

        for edge in edges:
            if not isinstance(edge, dict):
                print(f"Invalid edge entry in material '{material_name}': {edge}")
                continue

            # Extract types and convert to IDs
            from_type = edge.get("from_type")
            to_type = edge.get("to_type")

            from_id = node_to_id.get(from_type)
            to_id = node_to_id.get(to_type)

            # Only include valid edges
            if from_id is not None and to_id is not None:
                preprocessed_data.append({
                    "from": from_id,
                    "to": to_id
                })

    # Save the preprocessed data
    with open(output_file, "w") as file:
        json.dump(preprocessed_data, file, indent=4)

    print(f"Preprocessed data saved to '{output_file}'.")

# Run preprocessing
if __name__ == "__main__":
    preprocess_edge_database(edge_database_path, preprocessed_data_path, node_to_id_path)
