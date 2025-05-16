import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import json
from gnn_edge_predictor import GNNModel, NUM_NODE_TYPES, NUM_SOCKET_TYPES

# Load mappings
with open("/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json", "r") as f:
    id_to_node = json.load(f)
with open("/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json", "r") as f:
    id_to_socket = json.load(f)

# Compute correct max ID
NUM_NODE_TYPES = max(map(int, id_to_node.keys())) + 1
NUM_SOCKET_TYPES = max(map(int, id_to_socket.keys())) + 1

node_names = {int(k): v for k, v in id_to_node.items()}
socket_names = {int(k): v for k, v in id_to_socket.items()}

# Example input sequence
sequence = [58, 58, 58, 33, 56, 58, 43, 9, 23, 48, 41, 58, 17, 35]

node_types = torch.tensor(sequence, dtype=torch.long)
x = F.one_hot(node_types, NUM_NODE_TYPES).float()

edge_index = [
    [i, j] for i in range(len(sequence)) for j in range(len(sequence)) if i != j
]
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# New required edge features
edge_type_pair = torch.tensor([
    [sequence[src], sequence[dst]] for src, dst in edge_index.t().tolist()
], dtype=torch.long)

edge_distance = torch.tensor([
    [abs(src - dst)] for src, dst in edge_index.t().tolist()
], dtype=torch.float32)

# Dummy placeholder edge_attr and existence
edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.long)
edge_exists = torch.zeros((edge_index.size(1),), dtype=torch.float32)
socket_mask = torch.zeros_like(edge_exists)

data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    edge_exists=edge_exists,
    socket_mask=socket_mask,
    edge_type_pair=edge_type_pair,
    edge_distance=edge_distance
)

# Load model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = GNNModel(input_dim=NUM_NODE_TYPES, hidden_dim=64).to(device)
model.load_state_dict(torch.load("gnn_edge_model.pt", map_location=device))
model.eval()

# Predict
data = data.to(device)
with torch.no_grad():
    edge_logits, socket_pred = model(data)
    edge_probs = torch.sigmoid(edge_logits)
    socket_pred = socket_pred.view(-1, 2, NUM_SOCKET_TYPES)
    src_sock = socket_pred[:, 0, :].argmax(dim=1).cpu().tolist()
    dst_sock = socket_pred[:, 1, :].argmax(dim=1).cpu().tolist()
    edge_probs = edge_probs.cpu().tolist()
    edge_list = data.edge_index.t().cpu().tolist()

# Display
print("\nPredicted Edges with Sockets (prob > 0.95):")
for idx, (src, dst) in enumerate(edge_list):
    if edge_probs[idx] > 0.85:
        print(f"{src} ({node_names.get(sequence[src], 'UNKNOWN')}) --[{socket_names.get(src_sock[idx], 'UNKNOWN')}]--> "
              f"{dst} ({node_names.get(sequence[dst], 'UNKNOWN')}) [{socket_names.get(dst_sock[idx], 'UNKNOWN')}]  (score={edge_probs[idx]:.2f})")
