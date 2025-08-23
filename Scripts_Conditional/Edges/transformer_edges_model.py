import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
database_path = r"C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\merged_database.json"
model_save_path = r"C:\Users\hyperbook\Desktop\PPMGR\trained_edge_model"
edge_predictions_path = r"C:\Users\hyperbook\Desktop\PPMGR\predicted_edges.json"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Dataset Preparation
def preprocess_edge_dataset(input_file):
    logging.info("Loading node and edge data...")
    with open(input_file, "r") as file:
        data = json.load(file)

    node_features = []
    edge_connections = []

    for material_name, material_data in data["materials"].items():
        nodes = material_data["nodes"]
        edges = material_data["edges"]

        # Create a node feature list (node_type is used as input)
        node_features.append([node["type"] for node in nodes])

        # Create edge connections (as pairs of node indices)
        edge_connections.append([(edge["from_node"], edge["to_node"]) for edge in edges])

    return node_features, edge_connections

class EdgePredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EdgePredictionModel, self).__init__()
        self.encoder = nn.Embedding(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=4,  # Correct argument for the number of attention heads
            num_encoder_layers=2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features):
        # Encode nodes
        encoded_nodes = self.encoder(node_features)

        # Transformer forward pass
        transformer_output = self.transformer(encoded_nodes.unsqueeze(1), encoded_nodes.unsqueeze(1))

        # Linear layer for edge prediction
        edge_scores = self.fc(transformer_output.squeeze(1))
        return edge_scores

# Train the Edge Prediction Model
def train_edge_model(node_features, edge_connections, vocab_size):
    # Hyperparameters
    hidden_dim = 128
    output_dim = 1  # Binary classification (edge exists or not)
    batch_size = 32
    epochs = 10
    learning_rate = 1e-4

    # Prepare the model, loss, and optimizer
    model = EdgePredictionModel(vocab_size, hidden_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (node_feat, edge_conn) in enumerate(zip(node_features, edge_connections)):
            node_feat = torch.tensor(node_feat, dtype=torch.long).to(device)
            edge_conn = torch.tensor(edge_conn, dtype=torch.float32).to(device)

            optimizer.zero_grad()

            # Forward pass
            edge_scores = model(node_feat)

            # Calculate loss
            loss = criterion(edge_scores.view(-1), edge_conn.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the model
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_path, "edge_model.pt"))
    logging.info("Edge prediction model trained and saved.")

# Predict Edges
def predict_edges(node_features, vocab_size):
    model = EdgePredictionModel(vocab_size, hidden_dim=128, output_dim=1).to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_path, "edge_model.pt")))
    model.eval()

    predicted_edges = []

    with torch.no_grad():
        for node_feat in node_features:
            node_feat = torch.tensor(node_feat, dtype=torch.long).to(device)
            edge_scores = model(node_feat)
            edge_predictions = (edge_scores > 0.5).float()  # Threshold for edge existence
            predicted_edges.append(edge_predictions.cpu().numpy().tolist())

    with open(edge_predictions_path, "w") as file:
        json.dump(predicted_edges, file, indent=4)

    logging.info(f"Predicted edges saved to '{edge_predictions_path}'")

# Example Usage
if __name__ == "__main__":
    node_features, edge_connections = preprocess_edge_dataset(database_path)

    # Train the model (if required)
    train_edge_model(node_features, edge_connections, vocab_size=100)

    # Predict edges
    predict_edges(node_features, vocab_size=100)
