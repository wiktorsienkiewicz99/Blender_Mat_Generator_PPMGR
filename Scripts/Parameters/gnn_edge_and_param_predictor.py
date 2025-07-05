# model_and_train.py

import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

# ─────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[+] Using device: {device}")

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────
def parse_blender_vector(value_str):
    if not isinstance(value_str, str):
        return None
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", value_str)
    if matches:
        return [float(v) for v in matches]
    return None

# ─────────────────────────────────────────────
# STAT EXTRACTION: RANGES + CLASS MAPS
# ─────────────────────────────────────────────
def compute_param_stats(json_path):
    param_stats = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
    dropdown_candidates = defaultdict(set)
    boolean_candidates = defaultdict(set)

    with open(json_path, "r") as f:
        data = json.load(f)

    for mat in data["materials"].values():
        for node in mat["nodes"]:
            for inp in node.get("inputs", []):
                if not inp.get("is_linked"):
                    val = inp.get("value")
                    parsed = parse_blender_vector(val)
                    if parsed:
                        for i, v in enumerate(parsed):
                            key_i = f"{inp['name']}.{i}"
                            param_stats[key_i]["min"] = min(param_stats[key_i]["min"], v)
                            param_stats[key_i]["max"] = max(param_stats[key_i]["max"], v)
                    elif isinstance(val, (int, float)):
                        key = inp["name"]
                        param_stats[key]["min"] = min(param_stats[key]["min"], val)
                        param_stats[key]["max"] = max(param_stats[key]["max"], val)
                    elif isinstance(val, bool):
                        boolean_candidates[inp["name"]].add(val)

            for k, v in node.get("parameters", {}).items():
                if isinstance(v, (int, float)):
                    param_stats[k]["min"] = min(param_stats[k]["min"], v)
                    param_stats[k]["max"] = max(param_stats[k]["max"], v)
                elif isinstance(v, str) and v.strip() and k not in ["Image Name", "Image Path"]:
                    dropdown_candidates[k].add(v)
                elif isinstance(v, bool):
                    boolean_candidates[k].add(v)

    param_ranges = {k: (v["min"], v["max"]) for k, v in param_stats.items()}
    dropdown_classes = {k: sorted(list(v)) for k, v in dropdown_candidates.items() if len(v) > 1}
    checkbox_classes = {k: [False, True] for k, v in boolean_candidates.items() if len(v) > 0}
    return param_ranges, dropdown_classes, checkbox_classes

# ─────────────────────────────────────────────
# PARAM DATASET
# ─────────────────────────────────────────────
class ParamDataset(Dataset):
    def __init__(self, json_path, param_ranges, dropdown_classes, checkbox_classes, node_type_to_params):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.samples = []
        self.param_ranges = param_ranges
        self.dropdown_classes = dropdown_classes
        self.checkbox_classes = checkbox_classes
        self.node_type_to_params = node_type_to_params

        self.node_types = sorted({
            node["type"]
            for mat in data["materials"].values()
            for node in mat["nodes"]
        })
        self.node_type_list = self.node_types

        all_keys = set(param_ranges.keys()) | set(dropdown_classes.keys()) | set(checkbox_classes.keys())
        self.param_keys = sorted(list(all_keys))
        self.param_index = {k: i for i, k in enumerate(self.param_keys)}

        for mat in data["materials"].values():
            for node in mat["nodes"]:
                node_type = node["type"]
                valid_params = set(node_type_to_params.get(node_type, []))
                x = self.encode_input(node_type)

                for inp in node.get("inputs", []):
                    if not inp.get("is_linked"):
                        key = inp["name"]
                        val = inp.get("value")
                        parsed = parse_blender_vector(val)
                        if parsed:
                            for i, v in enumerate(parsed):
                                key_i = f"{key}.{i}"
                                if key_i in valid_params and key_i in param_ranges:
                                    self.samples.append({"x": x, "param_key": key_i, "y": self.normalize(v, key_i), "type": "reg"})
                        elif isinstance(val, (int, float)) and key in valid_params and key in param_ranges:
                            self.samples.append({"x": x, "param_key": key, "y": self.normalize(val, key), "type": "reg"})
                        elif isinstance(val, bool) and key in valid_params and key in checkbox_classes:
                            self.samples.append({"x": x, "param_key": key, "y": int(val), "type": "bin"})

                for k, v in node.get("parameters", {}).items():
                    if k in ["Image Name", "Image Path"]:
                        continue
                    if k not in valid_params:
                        continue
                    if isinstance(v, (int, float)) and k in param_ranges:
                        self.samples.append({"x": x, "param_key": k, "y": self.normalize(v, k), "type": "reg"})
                    elif isinstance(v, str) and k in dropdown_classes:
                        y = dropdown_classes[k].index(v)
                        self.samples.append({"x": x, "param_key": k, "y": y, "type": "cls"})
                    elif isinstance(v, bool) and k in checkbox_classes:
                        self.samples.append({"x": x, "param_key": k, "y": int(v), "type": "bin"})

        self.param_types = {s["param_key"]: s["type"] for s in self.samples}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return torch.tensor(s["x"], dtype=torch.float32), torch.tensor(s["y"]), self.param_index[s["param_key"]], s["param_key"], s["type"]

    def encode_input(self, node_type):
        return [1.0 if node_type == t else 0.0 for t in self.node_type_list]

    def normalize(self, value, key):
        min_val, max_val = self.param_ranges[key]
        return (value - min_val) / (max_val - min_val + 1e-8)

# ─────────────────────────────────────────────
# Multi-head Model
# ─────────────────────────────────────────────

class MultiHeadParamPredictor(nn.Module):
    def __init__(self, input_dim, param_keys, param_types, dropdown_classes, checkbox_classes):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.param_keys = param_keys
        self.param_types = param_types

        # Replace '.' with '__' for safe PyTorch keys
        self.key_map = {k: k.replace(".", "__") for k in param_keys}
        self.reverse_key_map = {v: k for k, v in self.key_map.items()}

        self.regression_heads = nn.ModuleDict()
        self.classification_heads = nn.ModuleDict()
        self.binary_heads = nn.ModuleDict()

        for key in param_keys:
            safe_key = self.key_map[key]
            param_type = param_types.get(key)
            if not param_type:
                continue
            if param_type == "reg":
                self.regression_heads[safe_key] = nn.Linear(64, 1)
            elif param_type == "cls":
                num_classes = len(dropdown_classes[key])
                self.classification_heads[safe_key] = nn.Linear(64, num_classes)
            elif param_type == "bin":
                self.binary_heads[safe_key] = nn.Linear(64, 2)

    def forward(self, x, param_keys):
        shared = self.shared(x)
        outputs = []

        for i, key in enumerate(param_keys):
            key_str = key if isinstance(key, str) else key.item()
            param_type = self.param_types.get(key_str)

            if not param_type:
                dummy = torch.zeros(1, requires_grad=True).to(x.device)
                outputs.append(dummy)
                continue

            safe_key = self.key_map.get(key_str)
            if param_type == "reg":
                outputs.append(self.regression_heads[safe_key](shared[i]))
            elif param_type == "cls":
                outputs.append(self.classification_heads[safe_key](shared[i]))
            elif param_type == "bin":
                outputs.append(self.binary_heads[safe_key](shared[i]))
            else:
                dummy = torch.zeros(1, requires_grad=True).to(x.device)
                outputs.append(dummy)

        return outputs
# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train(model, dataloader, param_types, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn_reg = nn.MSELoss()
    loss_fn_cls = nn.CrossEntropyLoss()
    loss_fn_bin = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y, param_idx, param_key, param_type in dataloader:
            preds = model(x, param_key)
            losses = []
            for i in range(len(preds)):
                if param_type[i] == "reg":
                    losses.append(loss_fn_reg(preds[i].squeeze(), y[i]))
                elif param_type[i] == "cls":
                    losses.append(loss_fn_cls(preds[i].unsqueeze(0), y[i].long().unsqueeze(0)))
                elif param_type[i] == "bin":
                    losses.append(loss_fn_bin(preds[i].unsqueeze(0), y[i].long().unsqueeze(0)))
            loss = sum(losses) / len(losses)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    JSON_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/merged_dataset.json"
    NODE_TYPE_TO_PARAMS_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_params.json"
    MODEL_SAVE_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Parameters/param_predictor.pth"

    with open(NODE_TYPE_TO_PARAMS_PATH) as f:
        node_type_to_params = json.load(f)

    param_ranges, dropdown_classes, checkbox_classes = compute_param_stats(JSON_PATH)
    dataset = ParamDataset(JSON_PATH, param_ranges, dropdown_classes, checkbox_classes, node_type_to_params)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiHeadParamPredictor(
        input_dim=len(dataset[0][0]),
        param_keys=dataset.param_keys,
        param_types=dataset.param_types,
        dropdown_classes=dropdown_classes,
        checkbox_classes=checkbox_classes
    )

    train(model, dataloader, dataset.param_types, epochs=10)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[+] Saved model to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()