import argparse
import json
import os
import pickle
from typing import Dict, List

import torch

from gnn_edge_and_param_predictor import (
    compute_param_stats,
    ParamDataset,
    MultiHeadParamPredictor,
)


def parse_args() -> argparse.Namespace:
    """Command line arguments."""
    p = argparse.ArgumentParser(description="Predict node parameters")
    p.add_argument("--param-model", required=True, help="Path to parameter model")
    p.add_argument("--param-json", required=True, help="Path to parameter dataset JSON")
    p.add_argument(
        "--node-type-to-params", required=True, help="Mapping of node types to parameters"
    )
    p.add_argument("--id2node-json", required=True, help="id_to_node mapping JSON")
    p.add_argument(
        "--graph-json",
        required=True,
        help="Predicted material graph containing 'node_sequence'",
    )
    p.add_argument(
        "--material-name",
        default="Generated_Material",
        help="Material name used as context",
    )
    p.add_argument(
        "--use-material-context",
        action="store_true",
        help="Use material name context if supported",
    )
    p.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save graph with predicted parameters",
    )
    return p.parse_args()


def encode_param_key(key: str) -> str:
    """Make parameter keys safe for PyTorch modules."""
    return key.replace(".", "__")


def load_model(
    model_path: str,
    dummy_dataset: ParamDataset,
    param_types: Dict[str, str],
    dropdown: Dict[str, List[str]],
    checkbox: Dict[str, List[bool]],
    use_context: bool,
    device: torch.device,
) -> MultiHeadParamPredictor:
    """Load the parameter model with optional material context."""
    state = torch.load(model_path, map_location=device)
    has_meta = isinstance(state, dict) and "model_state_dict" in state
    material_dim = state.get("material_feature_dim") if has_meta and use_context else None
    model = MultiHeadParamPredictor(
        input_dim=len(dummy_dataset[0][0]),
        param_keys=[encode_param_key(k) for k in dummy_dataset.param_keys],
        param_types={encode_param_key(k): v for k, v in param_types.items()},
        dropdown_classes={encode_param_key(k): v for k, v in dropdown.items()},
        checkbox_classes={encode_param_key(k): v for k, v in checkbox.items()},
        material_feature_dim=material_dim,
    )
    model.load_state_dict(state["model_state_dict"] if has_meta else state)
    model.to(device).eval()
    return model


def get_material_features(model_path: str, material_name: str, device: torch.device):
    """Load TF-IDF vectorizer and encode the material name."""
    vec_path = os.path.join(os.path.dirname(model_path), "material_name_vectorizer.pkl")
    if not os.path.exists(vec_path):
        return None
    try:
        with open(vec_path, "rb") as f:
            vec = pickle.load(f)
        feats = vec.transform([material_name])
        return torch.tensor(feats.toarray(), dtype=torch.float32).squeeze(0).to(device)
    except Exception as exc:
        print(f"[!] Failed loading material context: {exc}")
        return None


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load mappings and dataset statistics
    param_ranges, dropdown, checkbox = compute_param_stats(args.param_json)
    with open(args.node_type_to_params) as f:
        node_type_to_params = json.load(f)
    with open(args.id2node_json) as f:
        id_to_node = {int(k): v for k, v in json.load(f).items()}
    with open(args.graph_json) as f:
        graph = json.load(f)
    node_sequence = graph["node_sequence"]

    dataset = ParamDataset(args.param_json, param_ranges, dropdown, checkbox, node_type_to_params)
    param_types = dataset.param_types

    model = load_model(
        args.param_model,
        dataset,
        param_types,
        dropdown,
        checkbox,
        args.use_material_context,
        device,
    )

    material_feat = None
    if args.use_material_context and getattr(model, "has_material_context", False):
        material_feat = get_material_features(args.param_model, args.material_name, device)
        if material_feat is None:
            print("[!] Material context requested but vectorizer missing; continuing without context")

    print("\n🔍 Predicting parameter values for each node:")
    graph.setdefault("parameters", {})

    param_keys = list(model.param_types.keys())
    EXCLUDED = {"Image Name", "Image Path"}
    for idx, node_id in enumerate(node_sequence):
        node_type = id_to_node.get(node_id, "Unknown")
        one_hot = [1.0 if t == node_type else 0.0 for t in dataset.node_type_list]
        x = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0).to(device)

        if material_feat is not None:
            outputs = model(x, param_keys, material_feat.unsqueeze(0))
            print(f"\nNode {idx} [{node_type}] (context: {args.material_name}):")
        else:
            outputs = model(x, param_keys)
            print(f"\nNode {idx} [{node_type}]:")

        graph["parameters"][idx] = {"node_type": node_type, "params": {}}
        for out, key in zip(outputs, param_keys):
            orig = key.replace("__", ".")
            if orig not in node_type_to_params.get(node_type, []) or orig in EXCLUDED:
                continue
            typ = model.param_types[key]
            if typ == "reg":
                val = out.squeeze().item()
                denorm = val * (param_ranges[orig][1] - param_ranges[orig][0] + 1e-8) + param_ranges[orig][0]
                graph["parameters"][idx]["params"][orig] = denorm
                print(f"  {orig:<20} = {denorm:.4f}")
            elif typ == "cls":
                logits = out.squeeze()
                cls = int(torch.argmax(logits))
                label = dropdown[key][cls]
                graph["parameters"][idx]["params"][orig] = label
                print(f"  {orig:<20} = '{label}'")
            elif typ == "bin":
                logits = out.squeeze()
                cls = int(torch.argmax(logits))
                label = model.checkbox_classes[key][cls]
                graph["parameters"][idx]["params"][orig] = label
                print(f"  {orig:<20} = {label}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(graph, f, indent=2)
        print(f"\n✅ Saved graph with parameters to {args.output_json}")


if __name__ == "__main__":
    main()
