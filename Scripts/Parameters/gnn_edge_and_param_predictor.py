import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ─────────────────────────────────────────────
# Auto-Detect Normalization Ranges Per Key
# ─────────────────────────────────────────────

def compute_param_stats(json_path):
    param_stats = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})

    with open(json_path, "r") as f:
        data = json.load(f)

    for mat in data["materials"].values():
        for node in mat["nodes"]:
            for inp in node.get("inputs", []):
                if not inp.get("is_linked") and isinstance(inp.get("value"), (int, float)):
                    key = inp["name"]
                    val = inp["value"]
                    param_stats[key]["min"] = min(param_stats[key]["min"], val)
                    param_stats[key]["max"] = max(param_stats[key]["max"], val)
            for k, v in node.get("parameters", {}).items():
                if isinstance(v, (int, float)):
                    param_stats[k]["min"] = min(param_stats[k]["min"], v)
                    param_stats[k]["max"] = max(param_stats[k]["max"], v)

    return {k: (v["min"], v["max"]) for k, v in param_stats.items()}

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ParamDataset(Dataset):
    def __init__(self, json_path, param_ranges):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.samples = []
        self.node_types = set()
        self.param_keys = set()
        self.param_ranges = param_ranges

        for mat in data["materials"].values():
            for node in mat["nodes"]:
                self.node_types.add(node["type"])

        self.node_type_list = sorted(list(self.node_types))

        for mat in data["materials"].values():
            for node in mat["nodes"]:
                for inp in node.get("inputs", []):
                    if not inp.get("is_linked") and isinstance(inp.get("value"), (int, float)):
                        self.param_keys.add(inp["name"])
                for k, v in node.get("parameters", {}).items():
                    if isinstance(v, (int, float)):
                        self.param_keys.add(k)

        self.param_key_list = sorted(list(self.param_keys))
        self.param_index = {k: i for i, k in enumerate(self.param_key_list)}

        for mat in data["materials"].values():
            for node in mat["nodes"]:
                node_type = node["type"]
                for inp in node.get("inputs", []):
                    key = inp["name"]
                    val = inp.get("value")
                    if not inp.get("is_linked") and isinstance(val, (int, float)):
                        if key not in self.param_ranges:
                            continue
                        x = self.encode_input(node_type)
                        self.samples.append({
                            "x": x,
                            "param_key": key,
                            "y": self.normalize(val, key)
                        })

                for k, v in node.get("parameters", {}).items():
                    if isinstance(v, (int, float)):
                        if k not in self.param_ranges:
                            continue
                        x = self.encode_input(node_type)
                        self.samples.append({
                            "x": x,
                            "param_key": k,
                            "y": self.normalize(v, k)
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        x = torch.tensor(item["x"], dtype=torch.float32)
        y = torch.tensor(item["y"], dtype=torch.float32)
        target_index = self.param_index[item["param_key"]]
        return x, y, target_index, item["param_key"]

    def encode_input(self, node_type):
        return [1.0 if node_type == t else 0.0 for t in self.node_type_list]

    def normalize(self, value, key):
        min_val, max_val = self.param_ranges[key]
        return (value - min_val) / (max_val - min_val + 1e-8)

# ─────────────────────────────────────────────
# Multi-head Model
# ─────────────────────────────────────────────

class MultiHeadParamPredictor(nn.Module):
    def __init__(self, input_dim, num_params):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in range(num_params)])

    def forward(self, x, param_index):
        base = self.shared(x)
        out = torch.stack([head(base).squeeze(1) for head in self.heads], dim=1)
        return torch.gather(out, 1, param_index.unsqueeze(1)).squeeze(1)

# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(model, dataloader, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (x, y, param_idx, param_key) in enumerate(dataloader):
            pred = model(x, param_idx)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch == 0 and i == 0:
                print("\n[DEBUG] First batch:")
                for j in range(min(5, len(pred))):
                    print(f"Param: {param_key[j]:>12} | Target: {y[j]:.4f} | Predicted: {pred[j]:.4f}")

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    JSON_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/merged_dataset.json"
    param_ranges = compute_param_stats(JSON_PATH)
    dataset = ParamDataset(JSON_PATH, param_ranges)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = len(dataset[0][0])
    num_params = len(dataset.param_key_list)
    print(f"\nTraining on {len(dataset)} samples | Input dim: {input_dim} | Param heads: {num_params}")

    model = MultiHeadParamPredictor(input_dim, num_params)
    train(model, dataloader, epochs=25)

if __name__ == "__main__":
    main()