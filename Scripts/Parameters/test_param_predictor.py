import torch
import json
from gnn_edge_and_param_predictor import compute_param_stats, ParamDataset, MultiHeadParamPredictor

# ─────────────────────────────
# CONFIGURATION
# ─────────────────────────────
MODEL_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Parameters/param_predictor.pth"
ID_TO_NODE_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json"
PARAM_JSON_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/merged_dataset.json"
PREDICTED_GRAPH_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Generated/predicted_material_graph.json"
NODE_TYPE_TO_PARAMS_PATH = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_params.json"

EXCLUDED_PARAMS = {"Image Name", "Image Path"}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[+] Using device: {device}")

# ─────────────────────────────
# LOAD STRUCTURE + MAPPINGS
# ─────────────────────────────
param_ranges, dropdown_classes, checkbox_classes = compute_param_stats(PARAM_JSON_PATH)

with open(NODE_TYPE_TO_PARAMS_PATH) as f:
    node_type_to_params = json.load(f)

with open(ID_TO_NODE_PATH) as f:
    id_to_node = {int(k): v for k, v in json.load(f).items()}

with open(PREDICTED_GRAPH_PATH) as f:
    data = json.load(f)
node_sequence = data["node_sequence"]

# Create dummy dataset for parameter metadata
dummy_dataset = ParamDataset(PARAM_JSON_PATH, param_ranges, dropdown_classes, checkbox_classes, node_type_to_params)
param_types = dummy_dataset.param_types

# ─────────────────────────────
# Helper: Encode/Decode Param Keys
# ─────────────────────────────
def encode_param_key(key: str) -> str:
    return key.replace(".", "__")

# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
model = MultiHeadParamPredictor(
    input_dim=len(dummy_dataset[0][0]),
    param_keys=[encode_param_key(k) for k in dummy_dataset.param_keys],
    param_types={encode_param_key(k): v for k, v in param_types.items()},
    dropdown_classes={encode_param_key(k): v for k, v in dropdown_classes.items()},
    checkbox_classes={encode_param_key(k): v for k, v in checkbox_classes.items()}
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# ─────────────────────────────
# PREDICT PARAMETERS PER NODE
# ─────────────────────────────
print("\n🔍 Predicting parameter values for each node:")

for idx, node_id in enumerate(node_sequence):
    node_type = id_to_node.get(node_id, "Unknown")
    one_hot = [1.0 if t == node_type else 0.0 for t in dummy_dataset.node_type_list]
    x = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0).to(device)

    shared = model.shared(x)
    print(f"\nNode {idx} [{node_type}]:")

    for param in node_type_to_params.get(node_type, []):
        if param in EXCLUDED_PARAMS:
            continue

        encoded = encode_param_key(param)
        head_type = model.param_types.get(encoded)
        if not head_type:
            continue

        if head_type == "reg" and encoded in model.regression_heads:
            pred = model.regression_heads[encoded](shared).squeeze().item()
            denorm = pred * (param_ranges[param][1] - param_ranges[param][0] + 1e-8) + param_ranges[param][0]
            print(f"  {param:<20} = {denorm:.4f}")

        elif head_type == "cls" and encoded in model.classification_heads:
            logits = model.classification_heads[encoded](shared).squeeze()
            class_idx = torch.argmax(logits).item()
            label = dropdown_classes[encoded][class_idx]
            print(f"  {param:<20} = '{label}'")

        elif head_type == "bin" and encoded in model.binary_heads:
            logits = model.binary_heads[encoded](shared).squeeze()
            class_idx = torch.argmax(logits).item()
            label = model.checkbox_classes[encoded][class_idx]
            print(f"  {param:<20} = {label}")