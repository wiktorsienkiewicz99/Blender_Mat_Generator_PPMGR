import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Function to extract node features from the JSON
def extract_node_features(node):
    return {
        "Type": node["Type"],
        "Parameters": node.get("Parameters", {})
    }

def extract_edges_from_tree(material):
    edges = []
    
    # Traverse the node tree to extract edges, skipping Group Input/Output nodes
    for node in material["Node Tree"]:
        node_name = node["Node"]
        node_type = node.get("Type", "")
        
        if node_type in ['GROUP_INPUT', 'GROUP_OUTPUT']:
            print(f"Skipping node: {node_name} of type {node_type}")  # Debugging output
            continue

        print(f"Processing node: {node_name}")  # Debugging output
        
        # Go through each output and its connections
        if "Outputs" in node:
            for output_name, connections in node["Outputs"].items():
                print(f"  Output: {output_name} has connections: {connections}")  # Debugging output

                # Case 1: If the connections are a list (valid links), process them
                if isinstance(connections, list) and connections:
                    for connection in connections:
                        target_node = connection["Target Node"]
                        target_socket = connection["Target Socket"]
                        print(f"    Connection: {node_name}.{output_name} -> {target_node}.{target_socket}")  # Debugging output
                        # Append the edge (connection) to the edges list
                        edges.append({
                            "Source": node_name,  # Use the full node name
                            "Source Socket": output_name,
                            "Target": target_node,  # Use the full target node name
                            "Target Socket": target_socket
                        })
                
                # Case 2: If the connections are a single dictionary
                elif isinstance(connections, dict):
                    target_node = connections["Target Node"]
                    target_socket = connections["Target Socket"]
                    print(f"    Single Connection: {node_name}.{output_name} -> {target_node}.{target_socket}")  # Debugging output
                    edges.append({
                        "Source": node_name,
                        "Source Socket": output_name,
                        "Target": target_node,
                        "Target Socket": target_socket
                    })

                # If there are no connections, skip
                elif connections == "No connections":
                    print(f"    No connections for output {output_name}")
    
    return edges

# Function to build the graph for a single material
def build_graph_for_material(material):
    graph = nx.DiGraph()

    # Add nodes with features, skipping Group Input/Output nodes
    for node in material["Node Tree"]:
        if node is not None and node.get("Type") not in ['GROUP_INPUT', 'GROUP_OUTPUT']:  # Ensure node is not Group Input/Output
            node_name = node["Node"]
            # Add each node with its unique name and attributes
            graph.add_node(node_name, **extract_node_features(node))

    # Add edges (linked node connections directly from the tree)
    edges = extract_edges_from_tree(material)
    for edge in edges:
        source_node = edge["Source"]
        target_node = edge["Target"]
        graph.add_edge(source_node, target_node)

    return graph

# Function to visualize the graph for a single material
def visualize_graph(graph, title="Material Node Graph"):
    pos = nx.spring_layout(graph)  # Define the layout for visualization

    plt.clf()  # Clear the previous plot
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', font_size=8, font_weight='bold', edge_color='gray')
    
    plt.title(title)
    plt.draw()  # Redraw the current plot

# Function to handle cycling through materials
class MaterialVisualizer:
    def __init__(self, json_data):
        self.materials = json_data
        self.current_index = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.build_visualization()
    
    def build_visualization(self):
        plt.subplots_adjust(bottom=0.2)

        # Add buttons for cycling through materials
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_material)
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_material)

        # Visualize the first material
        self.visualize_current_material()

    def visualize_current_material(self):
        material = self.materials[self.current_index]
        material_graph = build_graph_for_material(material)
        material_name = material["Material"]
        visualize_graph(material_graph, title=f"Material: {material_name}")

    def next_material(self, event):
        self.current_index = (self.current_index + 1) % len(self.materials)
        self.visualize_current_material()

    def prev_material(self, event):
        self.current_index = (self.current_index - 1) % len(self.materials)
        self.visualize_current_material()

# Function to read JSON file
def read_json_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Sample usage
input_file_path = r'C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\database\Mat_Tech_materials_database_test_group.json'

# Load the JSON data
input_data = read_json_from_file(input_file_path)

# Create a MaterialVisualizer instance to handle visualization
visualizer = MaterialVisualizer(input_data)

plt.show()
