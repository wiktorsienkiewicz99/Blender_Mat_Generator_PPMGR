import torch
import json
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import train_test_split_edges
import torch.nn as nn
import torch.nn.functional as F

# Load the database JSON file
database_path = "C:\\Users\\hyperbook\\Desktop\\PPMGR\\Projects\\Results\\database_merged.json"
with open(database_path, 'r') as f:
    database = json.load(f)

# Node type encoding
node_type_encoding = {
    "AMBIENT_OCCLUSION": 1,
    "BEVEL": 2,
    "BRIGHTCONTRAST": 3,
    "BSDF_DIFFUSE": 4,
    "BSDF_GLOSSY": 5,
    "BSDF_PRINCIPLED": 6,
    "BSDF_TRANSPARENT": 7,
    "BUMP": 8,
    "CLAMP": 9,
    "COMBINE_COLOR": 10,
    "COMBXYZ": 11,
    "CURVE_RGB": 12,
    "DISPLACEMENT": 13,
    "EMISSION": 14,
    "FRESNEL": 15,
    "GAMMA": 16,
    "GROUP": 17,
    "HUE_SAT": 18,
    "INVERT": 19,
    "LAYER_WEIGHT": 20,
    "LIGHT_PATH": 21,
    "MAPPING": 22,
    "MAP_RANGE": 23,
    "MATH": 24,
    "MIX": 25,
    "MIX_RGB": 26,
    "MIX_SHADER": 27,
    "NEW_GEOMETRY": 28,
    "NORMAL_MAP": 29,
    "OBJECT_INFO": 30,
    "OUTPUT_MATERIAL": 31,
    "RGBTOBW": 32,
    "SEPARATE_COLOR": 33,
    "SEPXYZ": 34,
    "SHADERTORGB": 35,
    "TEX_BRICK": 36,
    "TEX_CHECKER": 37,
    "TEX_COORD": 38,
    "TEX_GRADIENT": 39,
    "TEX_IMAGE": 40,
    "TEX_MAGIC": 41,
    "TEX_NOISE": 42,
    "TEX_VORONOI": 43,
    "TEX_WAVE": 44,
    "UVMAP": 45,
    "VALTORGB": 46,
    "VALUE": 47,
    "VECTOR_DISPLACEMENT": 48,
    "VECTOR_ROTATE": 49,
    "VECT_MATH": 50,
    "VECT_TRANSFORM": 51
}

# Sample function to encode each node with its features (e.g., node type, input/output link status)
def encode_node(node, max_inputs=10, max_outputs=5):
    # Node type embedding
    node_type = node_type_encoding.get(node["type"], -1)

    # Linked inputs encoding (1 if linked, 0 if not linked)
    linked_inputs = [1 if input_data["is_linked"] else 0 for input_data in node["inputs"]]
    linked_outputs = [1] * len(node["outputs"])  # Assuming outputs are always linked

    # Pad linked_inputs and linked_outputs to fixed lengths
    linked_inputs = linked_inputs[:max_inputs] + [0] * (max_inputs - len(linked_inputs))
    linked_outputs = linked_outputs[:max_outputs] + [0] * (max_outputs - len(linked_outputs))

    # Combine encodings into a fixed-length vector
    features = [node_type] + linked_inputs + linked_outputs
    return torch.tensor(features, dtype=torch.float)

# Function to prepare data
def prepare_data(database, max_inputs=10, max_outputs=5):
    nodes = database["nodes"]
    edges = database["edges"]

    # Create feature list and edge index
    node_features = []
    edge_index = []

    node_mapping = {}  # Mapping for node names to index
    idx = 0

    # Process nodes
    for material_name, node_list in nodes.items():
        for node in node_list:
            node_mapping[node["name"]] = idx
            node_features.append(encode_node(node, max_inputs, max_outputs))
            idx += 1

    # Process edges with existence check
    for material_name, edge_list in edges.items():
        for edge in edge_list:
            from_node = edge["from_node"]
            to_node = edge["to_node"]
            
            # Check if both nodes exist in the mapping
            if from_node in node_mapping and to_node in node_mapping:
                from_node_idx = node_mapping[from_node]
                to_node_idx = node_mapping[to_node]
                edge_index.append([from_node_idx, to_node_idx])
            else:
                # Log warning if the nodes are missing
                print(f"Warning: Edge references missing node(s): '{from_node}' or '{to_node}'")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features = torch.stack(node_features)

    return Data(x=node_features, edge_index=edge_index)

# Prepare data and transfer to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = prepare_data(database).to(device)


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv_mu = GCNConv(hidden_dim, out_dim)
        self.conv_logstd = GCNConv(hidden_dim, out_dim)  # Separate layer for log standard deviation

    def forward(self, x, edge_index):
        # Shared first layer
        x = F.relu(self.conv1(x, edge_index))
        # Separate layers for mu and logstd
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd

# Define Variational Graph Autoencoder
class MaterialGraphVGAE(VGAE):
    def __init__(self, encoder):
        super(MaterialGraphVGAE, self).__init__(encoder)

# Initialize model, optimizer, and loss function
input_dim = data.x.size(1)  # Node feature dimension
hidden_dim = 64
out_dim = 32  # Latent space dimension

encoder = GCNEncoder(input_dim, hidden_dim, out_dim)
model = MaterialGraphVGAE(encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Prepare data for training (train-test split)
data = train_test_split_edges(data)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss += (1 / data.num_nodes) * model.kl_loss()  # Add KL divergence for regularization
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
        return model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

# Training loop
for epoch in range(200):
    loss = train()
    auc, ap = test()
    print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')
