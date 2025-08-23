import json
import random
from itertools import combinations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv

# Paths
preprocessed_data_path = r"C:\Users\hyperbook\Desktop\PPMGR\preprocessed_edge_data.json"
model_save_path = r"C:\Users\hyperbook\Desktop\PPMGR\trained_sequence_edge_model.pt"

# Data Preparation
def prepare_training_data(input_file, max_nodes=10):
    """Prepare training data for the GNN."""
    with open(input_file, "r") as file:
        data = json.load(file)

    training_data = []

    for graph in data:
        sequence = graph["sequence"]
        edges = graph["edges"]

        # Generate training samples
        for _ in range(5):  # Generate multiple samples per graph
            random_nodes = random.sample(sequence, k=min(max_nodes, len(sequence)))
            valid_edges = [
                edge for edge in edges if edge["from"] in random_nodes and edge["to"] in random_nodes
            ]
            training_data.append({"nodes": random_nodes, "edges": valid_edges})

    return training_data


# Custom Dataset
class GraphDataset(Dataset):
    """Dataset for graph edge prediction."""
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph = self.data[idx]
        nodes = graph["nodes"]
        edges = graph["edges"]
        edge_set = {(edge["from"], edge["to"]) for edge in edges}
        return torch.tensor(nodes, dtype=torch.long), edge_set


# Model Definition
class GNNModel(nn.Module):
    """Graph Neural Network for edge prediction."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(GNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gcn1 = GCNConv(embed_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = self.embedding(x)
        x = self.gcn1(x, edge_index).relu()
        x = self.gcn2(x, edge_index).relu()
        return x

    def predict_edges(self, node_features, edge_candidates):
        """Predict probabilities for candidate edges."""
        edge_probs = []
        for u, v in edge_candidates:
            u_feat = node_features[u]
            v_feat = node_features[v]
            combined = torch.cat([u_feat, v_feat])
            prob = self.activation(self.fc(combined))
            edge_probs.append(prob)
        return torch.stack(edge_probs)


# Training
def train_gnn_model(dataset, vocab_size, embed_dim=16, hidden_dim=32, epochs=10, lr=0.001):
    """Train the GNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNModel(vocab_size, embed_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for nodes, edge_set in dataloader:
            nodes = nodes[0].to(device)
            optimizer.zero_grad()

            # Create all possible edges
            edge_candidates = list(combinations(range(len(nodes)), 2))
            edge_targets = torch.zeros(len(edge_candidates), dtype=torch.float, device=device)

            # Mark valid edges
            for i, (u, v) in enumerate(edge_candidates):
                if (nodes[u].item(), nodes[v].item()) in edge_set:
                    edge_targets[i] = 1.0

            # Predict edge probabilities
            node_features = model.embedding(nodes)
            edge_probs = model.predict_edges(node_features, edge_candidates)
            loss = loss_fn(edge_probs, edge_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model


# Prediction
def predict_graph(model, node_ids, vocab_size):
    """Predict a graph for the given node IDs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        x = torch.tensor(node_ids, dtype=torch.long).to(device)
        edge_candidates = list(combinations(range(len(node_ids)), 2))
        edge_probs = model.predict_edges(model.embedding(x), edge_candidates)
        predicted_edges = [
            (node_ids[u], node_ids[v])
            for i, (u, v) in enumerate(edge_candidates)
            if edge_probs[i] > 0.5
        ]
    return predicted_edges


# Main Script
if __name__ == "__main__":
    # Prepare Data
    training_data = prepare_training_data(preprocessed_data_path)
    vocab_size = max(node for graph in training_data for node in graph["nodes"]) + 1
    dataset = GraphDataset(training_data)

    # Train Model
    model = train_gnn_model(dataset, vocab_size)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}.")

    # Test Prediction
    test_nodes = [13, 22, 9, 1, 11]  # Example input
    predicted_edges = predict_graph(model, test_nodes, vocab_size)
    print(f"Predicted edges: {predicted_edges}")
