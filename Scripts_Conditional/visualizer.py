import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Load JSON data
with open(r'C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\database_merged.json') as f:
    data = json.load(f)

# Initialize variables
material_names = list(data['nodes'].keys())  # Get the list of materials
material_index = 0  # Start with the first material

# Function to draw the graph for a specific material
def draw_graph(material_name):
    plt.clf()  # Clear the current figure
    G = nx.DiGraph()

    # Add nodes for the current material
    for node in data['nodes'][material_name]:
        node_name = node['name']
        node_type = node['type']
        G.add_node(node_name, label=node_type)

    # Add edges for the current material
    # Handle potential nesting in the edges structure
    if 'edges' in data and material_name in data['edges']:
        edges = data['edges'][material_name]
        if isinstance(edges, list):  # Check if edges are directly in a list
            for edge in edges:
                from_node = edge['from_node']
                to_node = edge['to_node']
                G.add_edge(from_node, to_node)

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="lightblue", font_size=8, font_weight="bold", edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, font_size=7)
    plt.title(f"Nodes and Edges Graph for Material: {material_name}", fontsize=16, fontweight="bold")
    plt.draw()

# Button callback functions
def next_material(event):
    global material_index
    material_index = (material_index + 1) % len(material_names)
    draw_graph(material_names[material_index])

def prev_material(event):
    global material_index
    material_index = (material_index - 1) % len(material_names)
    draw_graph(material_names[material_index])

# Initialize plot and draw the first material
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
draw_graph(material_names[material_index])

# Create buttons for cycling through materials
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bprev = Button(axprev, 'Previous')
bnext.on_clicked(next_material)
bprev.on_clicked(prev_material)

plt.show()