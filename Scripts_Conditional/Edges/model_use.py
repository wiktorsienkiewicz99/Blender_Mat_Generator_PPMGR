import torch
import json

# Paths
model_path = r"C:\Users\hyperbook\Desktop\PPMGR\Trained_edges_model\trained_edge_model.pt"
node_to_id_path = r"C:\Users\hyperbook\Desktop\PPMGR\node_to_id.json"

class EdgePredictionModel(torch.nn.Module):
    """Model for predicting edges between nodes."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(EdgePredictionModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc1 = torch.nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, from_node, to_node):
        from_emb = self.embedding(from_node)
        to_emb = self.embedding(to_node)
        combined = torch.cat([from_emb, to_emb], dim=-1)
        hidden = torch.relu(self.fc1(combined))
        edge_prob = self.activation(self.fc2(hidden))
        return edge_prob

def load_model(model_path, vocab_size, embed_dim=16, hidden_dim=32):
    """Load the trained model."""
    model = EdgePredictionModel(vocab_size, embed_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_edge(model, from_node_id, to_node_id):
    """Predict the likelihood of an edge between two nodes."""
    with torch.no_grad():
        from_node = torch.tensor([from_node_id], dtype=torch.long)
        to_node = torch.tensor([to_node_id], dtype=torch.long)
        prob = model(from_node, to_node)
        return prob.item()

def main():
    # Load the node-to-ID mapping
    with open(node_to_id_path, "r") as file:
        node_to_id = json.load(file)

    # Determine vocabulary size
    vocab_size = max(node_to_id.values()) + 1

    # Load the model
    model = load_model(model_path, vocab_size)

    # Example usage
    print("Enter node pairs to predict edges. Type 'exit' to quit.")
    while True:
        try:
            input_str = input("Enter 'from_node_id to_node_id': ")
            if input_str.lower() == "exit":
                break

            from_node_id, to_node_id = map(int, input_str.split())
            prob = predict_edge(model, from_node_id, to_node_id)
            print(f"Probability of edge between {from_node_id} and {to_node_id}: {prob:.4f}")
        except Exception as e:
            print(f"Error: {e}. Please try again.")

if __name__ == "__main__":
    main()
