import json

# Paths
database_path = r"C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\merged_database.json"
node_to_id_path = r"C:\Users\hyperbook\Desktop\PPMGR\node_to_id.json"
output_path = r"C:\Users\hyperbook\Desktop\PPMGR\cleaned_edge_data.json"

def clean_database(database_path, node_to_id_path, output_path):
    """Clean the database and generate sequence and edges using node type IDs."""
    # Load database and node-to-ID mapping
    with open(database_path, "r") as db_file:
        database = json.load(db_file)

    with open(node_to_id_path, "r") as id_file:
        node_to_id = json.load(id_file)

    cleaned_data = []

    # Process each material
    for material_name, material_data in database.get("materials", {}).items():
        nodes = material_data.get("nodes", [])
        edges = material_data.get("edges", [])

        # Create sequence from node types
        sequence = []
        node_name_to_id = {}

        for node in nodes:
            if isinstance(node, dict):  # Check if the node is a dictionary
                node_type = node.get("type")
                node_name = node.get("name")
            elif isinstance(node, str):  # If the node is a string, treat as name without type
                node_type = None
                node_name = node
            else:
                continue  # Skip invalid node entries

            node_id = node_to_id.get(node_type)
            if node_name and node_id is not None:
                sequence.append(node_id)
                node_name_to_id[node_name] = node_id

        # Convert edges to use IDs
        converted_edges = []
        for edge in edges:
            if isinstance(edge, dict):  # Ensure the edge is a dictionary
                from_node = edge.get("from_node")
                to_node = edge.get("to_node")

                from_id = node_name_to_id.get(from_node)
                to_id = node_name_to_id.get(to_node)

                if from_id is not None and to_id is not None:
                    converted_edges.append({"from": from_id, "to": to_id})
            else:
                print(f"Invalid edge entry in material '{material_name}': {edge}")
                continue

        # Add cleaned material data
        if sequence and converted_edges:
            cleaned_data.append({
                "material_name": material_name,
                "sequence": sorted(sequence),  # Sorted for consistency
                "edges": converted_edges
            })

    # Save the cleaned data
    with open(output_path, "w") as output_file:
        json.dump(cleaned_data, output_file, indent=4)

    print(f"Cleaned data saved to '{output_path}'.")

# Run the script
if __name__ == "__main__":
    clean_database(database_path, node_to_id_path, output_path)
