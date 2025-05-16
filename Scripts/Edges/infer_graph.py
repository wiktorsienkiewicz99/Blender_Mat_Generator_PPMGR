'''
Sample 4: TEX_IMAGE → TEX_IMAGE → TEX_IMAGE → MAPPING → TEX_COORD → TEX_IMAGE → OUTPUT_MATERIAL → BSDF_PRINCIPLED → DISPLACEMENT → SEPARATE_COLOR → NORMAL_MAP → TEX_IMAGE → COMBINE_COLOR → MATH
 [77, 58, 58, 58, 33, 56, 58, 43, 9, 23, 48, 41, 58, 17, 35, 78]



python infer_graph.py \
  --edge-model /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/v2_edge_model.pth \
  --socket-model /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/sockets_model.pth \
  --id-to-node-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
  --nodes 58 58 58 33 56 58 43 9 23 48 41 58 17 35 \
  --id-to-socket-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json \
  --threshold 0.00001
'''

import json
import torch
import argparse
from edge_autoregression_train import EdgeExistencePredictor
from train_sockets import SocketClassifier
# ─── Load Utility ────────────────────────────────────────────────────────────

def load_models(edge_model_path, socket_model_path, device):
    edge_ckpt = torch.load(edge_model_path, map_location=device)
    socket_ckpt = torch.load(socket_model_path, map_location=device)

    edge_model = EdgeExistencePredictor(
        vocab_size=128,  # match training
        type_vocab_size=1000,
        d_model=256,
        nhead=4,
        nlayers=4,
        max_nodes=64
    ).to(device)
    edge_model.load_state_dict(edge_ckpt)
    edge_model.eval()

    socket_model = SocketClassifier(
        type_vocab_size=len(socket_ckpt['type_to_idx']),
        sock_vocab_size=len(socket_ckpt['sock_to_idx']),
        d_model=128
    ).to(device)
    socket_model.load_state_dict(socket_ckpt['model'])
    socket_model.eval()

    return edge_model, socket_model, socket_ckpt

# ─── Inference ───────────────────────────────────────────────────────────────

def predict_graph(node_ids, node_types, edge_model, socket_model, socket_meta, id_to_socket, threshold=0.5):
    device = next(edge_model.parameters()).device
    n = len(node_ids)

    node_tensor = torch.LongTensor(node_ids).unsqueeze(0).to(device)
    type_tensor = torch.LongTensor([hash(t) % 1000 for t in node_types]).unsqueeze(0).to(device)

    with torch.no_grad():
        edge_logits = edge_model(node_tensor, type_tensor)[0]
        edge_probs = torch.sigmoid(edge_logits)

    predicted_edges = []
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if edge_probs[i, j].item() > threshold:
                predicted_edges.append((i, j))

    type_to_idx = socket_meta['type_to_idx']
    idx_to_sock = socket_meta['idx_to_sock']
    sock_vocab_size = len(socket_meta['sock_to_idx'])

    edge_details = []
    with torch.no_grad():
        for i, j in predicted_edges:
            if i >= n or j >= n:
                continue

            src_id = node_ids[i]
            dst_id = node_ids[j]
            src_type = node_types[i]
            dst_type = node_types[j]

            if src_type not in type_to_idx or dst_type not in type_to_idx:
                continue

            pair_tensor = torch.LongTensor([
                [type_to_idx[src_type], type_to_idx[dst_type]]
            ]).to(device)

            logits = socket_model(pair_tensor)
            pred_src_sock = logits[:, :sock_vocab_size].argmax(dim=1).item()
            pred_dst_sock = logits[:, sock_vocab_size:].argmax(dim=1).item()

            src_sock_name = id_to_socket.get(pred_src_sock, f"{pred_src_sock}")
            dst_sock_name = id_to_socket.get(pred_dst_sock, f"{pred_dst_sock}")

            edge_details.append({
                "from_id": src_id,
                "from_type": src_type,
                "from_socket_id": pred_src_sock,
                "from_socket_name": src_sock_name,
                "to_id": dst_id,
                "to_type": dst_type,
                "to_socket_id": pred_dst_sock,
                "to_socket_name": dst_sock_name
            })

    return edge_details

# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-model", required=True)
    parser.add_argument("--socket-model", required=True)
    parser.add_argument("--id-to-node-json", required=True)
    parser.add_argument("--id-to-socket-json", required=True)
    parser.add_argument("--input-json", help="Optional file with 'nodes' and 'node_types'")
    parser.add_argument("--nodes", nargs='+', type=int, help="List of node IDs")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    edge_model, socket_model, socket_meta = load_models(args.edge_model, args.socket_model, device)

    with open(args.id_to_node_json) as f:
        id_to_node = {int(k): v for k, v in json.load(f).items()}

    with open(args.id_to_socket_json) as f:
        id_to_socket = {int(k): v for k, v in json.load(f).items()}

    if args.nodes:
        node_ids = args.nodes
        node_types = [id_to_node.get(n, "UNKNOWN") for n in node_ids]
    elif args.input_json:
        with open(args.input_json) as f:
            sample = json.load(f)
        node_ids = sample["nodes"]
        node_types = sample["node_types"]
    else:
        raise ValueError("Either --input-json or --nodes must be provided")

    result = predict_graph(node_ids, node_types, edge_model, socket_model, socket_meta, id_to_socket, args.threshold)
    print("\nPredicted Edges with Sockets:")
    for edge in result:
        print(f"{edge['from_id']} ({edge['from_type']}) --[{edge['from_socket_id']}:{edge['from_socket_name']}]--> {edge['to_id']} ({edge['to_type']}) --[{edge['to_socket_id']}:{edge['to_socket_name']}]")
