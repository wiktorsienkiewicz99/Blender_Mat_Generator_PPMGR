"""
GraphEdge Prediction with Socket Classification for Blender-style Material Graphs
Uses GCN layers and edge features (node type pair + distance) to learn realistic edge formation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split
import json
import random

# ──────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────
DATA_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_graph_dataset.json"
NODE_MAP_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json"
SOCKET_MAP_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json"

HIDDEN_DIM = 64
BATCH_SIZE = 64
NUM_WORKERS = 4
NUM_NEGATIVES = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
POS_WEIGHT = 5.0
PATIENCE = 10
MAX_EPOCHS = 100
STEP_SIZE = 10
GAMMA = 0.9

# ──────────────────────────────────────────────────────
# MAPPINGS
# ──────────────────────────────────────────────────────
with open(NODE_MAP_PATH, "r") as f:
    id_to_node = json.load(f)
with open(SOCKET_MAP_PATH, "r") as f:
    id_to_socket = json.load(f)

NUM_NODE_TYPES = len(id_to_node)
NUM_SOCKET_TYPES = len(id_to_socket)

# ──────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────
class GraphEdgeDataset(Dataset):
    """Dataset for loading material graphs and generating positive/negative edge samples."""
    def __init__(self, path: str, num_negatives: int = NUM_NEGATIVES):
        super().__init__()
        with open(path, 'r') as f:
            all_graphs = json.load(f)
            self.graphs = [g for g in all_graphs if len(g["nodes"]) <= 30]
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        node_types = torch.tensor(g["nodes"], dtype=torch.long)
        x = F.one_hot(node_types, NUM_NODE_TYPES).float()

        id_to_local = {nid: i for i, nid in enumerate(g["nodes"])}
        existing_edges = set(
            (id_to_local[src], id_to_local[dst])
            for src, _, dst, _ in g["edges"]
            if src in id_to_local and dst in id_to_local
        )

        edge_index, edge_attr, edge_exists = [], [], []
        socket_mask, edge_type_pair, edge_distance = [], [], []

        for src, src_sock, dst, dst_sock in g["edges"]:
            if src in id_to_local and dst in id_to_local:
                src_local = id_to_local[src]
                dst_local = id_to_local[dst]
                edge_index.append([src_local, dst_local])
                edge_attr.append([src_sock, dst_sock])
                edge_exists.append(1.0)
                socket_mask.append(1.0)
                edge_type_pair.append([node_types[src_local].item(), node_types[dst_local].item()])
                edge_distance.append(abs(src_local - dst_local))

        all_possible = [(i, j) for i in range(len(g["nodes"])) for j in range(len(g["nodes"])) if i != j and (i, j) not in existing_edges]
        sampled_neg = random.sample(all_possible, min(self.num_negatives, len(all_possible)))
        for src, dst in sampled_neg:
            edge_index.append([src, dst])
            edge_attr.append([0, 0])
            edge_exists.append(0.0)
            socket_mask.append(0.0)
            edge_type_pair.append([node_types[src].item(), node_types[dst].item()])
            edge_distance.append(abs(src - dst))

        return Data(
            x=x,
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.long),
            edge_exists=torch.tensor(edge_exists, dtype=torch.float32),
            socket_mask=torch.tensor(socket_mask, dtype=torch.float32),
            edge_type_pair=torch.tensor(edge_type_pair, dtype=torch.long),
            edge_distance=torch.tensor(edge_distance, dtype=torch.float32).unsqueeze(-1)
        )

# ──────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────
class GNNModel(nn.Module):
    """Graph Convolutional Network with edge feature conditioning."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)

        self.type_embedding = nn.Embedding(NUM_NODE_TYPES * NUM_NODE_TYPES, 32)
        self.distance_embedding = nn.Linear(1, 8)

        self.edge_classifier = nn.Linear(hidden_dim * 2 + 32 + 8, 1)
        self.socket_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32 + 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2 * NUM_SOCKET_TYPES)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.dropout(self.norm1(self.gcn1(x, edge_index)).relu())
        x = self.dropout(self.norm2(self.gcn2(x, edge_index)).relu())
        x = self.dropout(self.norm3(self.gcn3(x, edge_index)).relu())

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        edge_reps = torch.cat([x[src_nodes], x[dst_nodes]], dim=1)

        type_pair_ids = data.edge_type_pair[:, 0] * NUM_NODE_TYPES + data.edge_type_pair[:, 1]
        type_embed = self.type_embedding(type_pair_ids)
        dist_embed = self.distance_embedding(data.edge_distance)
        edge_input = torch.cat([edge_reps, type_embed, dist_embed], dim=1)

        edge_exists = self.edge_classifier(edge_input).squeeze(-1)
        socket_logits = self.socket_predictor(edge_input)
        return edge_exists, socket_logits

# ──────────────────────────────────────────────────────
# TRAINING FUNCTION
# ──────────────────────────────────────────────────────
def train(model, loader, optimizer, bce_loss, ce_loss, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        exists_pred, socket_pred = model(data)
        if torch.isnan(exists_pred).any() or torch.isnan(socket_pred).any():
            continue

        exists_pred = torch.clamp(exists_pred, -10.0, 10.0)
        y_exists = data.edge_exists
        y_socket = data.edge_attr
        mask = data.socket_mask.bool()

        pred = socket_pred.view(-1, 2, NUM_SOCKET_TYPES)
        src_pred = pred[:, 0, :][mask]
        dst_pred = pred[:, 1, :][mask]
        src_true = y_socket[:, 0][mask]
        dst_true = y_socket[:, 1][mask]

        loss = bce_loss(exists_pred, y_exists) + ce_loss(src_pred, src_true) + ce_loss(dst_pred, dst_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ──────────────────────────────────────────────────────
# TESTING FUNCTION
# ──────────────────────────────────────────────────────
def test(model, loader, device):
    model.eval()
    correct, total = 0, 0
    sock_correct, sock_total = 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            edge_logits, socket_pred = model(data)
            edge_probs = torch.sigmoid(edge_logits)
            pred_edges = edge_probs > 0.5
            correct += (pred_edges == data.edge_exists.bool()).sum().item()
            total += len(edge_probs)

            mask = data.socket_mask.bool()
            pred = socket_pred.view(-1, 2, NUM_SOCKET_TYPES)
            src_pred = pred[:, 0, :].argmax(dim=1)[mask]
            dst_pred = pred[:, 1, :].argmax(dim=1)[mask]
            src_true = data.edge_attr[:, 0][mask]
            dst_true = data.edge_attr[:, 1][mask]

            sock_correct += (src_pred == src_true).sum().item()
            sock_correct += (dst_pred == dst_true).sum().item()
            sock_total += 2 * mask.sum().item()

    edge_acc = correct / total
    socket_acc = sock_correct / sock_total if sock_total > 0 else 0.0
    print(f"Edge Accuracy: {edge_acc:.4f} | Socket Accuracy: {socket_acc:.4f}")
    return edge_acc, socket_acc

# ──────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(42)

    dataset = GraphEdgeDataset(DATA_PATH)
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = GNNModel(input_dim=NUM_NODE_TYPES, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=device))
    ce = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_socket_acc = 0.0
    streak = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train(model, train_loader, optimizer, bce, ce, device)
        edge_acc, sock_acc = test(model, test_loader, device)
        print(f"Epoch {epoch:02d} - Loss: {loss:.4f} - Edge Acc: {edge_acc:.4f} - Socket Acc: {sock_acc:.4f}")

        if sock_acc > best_socket_acc:
            best_socket_acc = sock_acc
            streak = 0
            torch.save(model.state_dict(), "test_gnn_edge_model.pt")
        else:
            streak += 1
            if streak >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
        scheduler.step()

if __name__ == "__main__":
    main()
