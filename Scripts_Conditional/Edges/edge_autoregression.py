'''
python edge_autoregression.py sample \
  --model-in /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/edge_model.pth \
  --node-sequence "43 9 58 23 58 58 58 41 33 56 29 21 67 17 48 30" \
  --socket-id-to-name-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json \
  --id-to-node-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
  --node-type-to-sockets-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_sockets.json \
  --vocab-size 128 \
  --max-nodes 256 \
  --socket-vocab-size 128 \
  --edge-threshold 0.01

'''
'''
python edge_autoregression.py train \
  --data-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_graph_dataset.json \
  --model-out /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/edge_model.pth \
  --id-to-socket-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json \
  --id-to-node-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
  --node-type-to-sockets-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_sockets.json \
  --vocab-size 128 \
  --socket-vocab-size 128 \
  --limit-samples 2000 \
  --batch-size 8 \
  --nhead 4 \
  --nlayers 4 \
  --d-model 256 \
  --lr 1e-4 \
  --max-nodes 256 \
  --epochs 5
'''
'''
python edge_autoregression.py validate \
  --dataset /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_graph_dataset.json \
  --id-to-node /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
  --id-to-socket /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json \
  --node-type-to-sockets /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_sockets.json
  '''

# Same imports & setup as before...
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ─── Dataset ────────────────────────────────────────────────────────────────

class EdgeGraphDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        node_tensor = torch.LongTensor(sample["nodes"])
        edge_tensor = torch.LongTensor(sample["edges"]) if sample["edges"] else torch.zeros((0, 4), dtype=torch.long)
        return node_tensor, edge_tensor

def collate_graphs(batch):
    node_seqs = [b[0] for b in batch]
    edge_seqs = [b[1] for b in batch]
    node_lens = torch.LongTensor([len(seq) for seq in node_seqs])
    padded_nodes = nn.utils.rnn.pad_sequence(node_seqs, batch_first=True, padding_value=0)
    return padded_nodes, node_lens, edge_seqs
# ─── Model ───────────────────────────────────────────────────────────────────

class EdgePredictor(nn.Module):
    def __init__(self, vocab_size, socket_vocab_size, d_model, nhead, nlayers, max_nodes):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_nodes, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model),
            num_layers=nlayers
        )
        self.mlp_edge = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * socket_vocab_size)
        )
        self.edge_exists = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, node_seq):
        B, T = node_seq.shape
        pos = self.pos_emb(torch.arange(T, device=node_seq.device).unsqueeze(0).expand(B, T))
        h = self.token_emb(node_seq) + pos
        encoded = self.encoder(h.transpose(0, 1)).transpose(0, 1)  # [B, T, D]

        all_pairs = []
        for b in range(B):
            vecs = encoded[b]
            n = (node_seq[b] > 0).sum().item()
            mask = ~torch.eye(n, dtype=torch.bool, device=node_seq.device)
            src, dst = mask.nonzero(as_tuple=True)
            pair_vecs = torch.cat([vecs[src], vecs[dst]], dim=1)
            all_pairs.append(pair_vecs)

        x = torch.cat(all_pairs, dim=0)
        return self.mlp_edge(x), self.edge_exists(x).squeeze(1)

# ─── Training ────────────────────────────────────────────────────────────────
# Updated train() to derive socket mappings from dataset instead of node_type_to_sockets.json

def train(args):
    import json, torch
    from torch.utils.data import Subset
    from torch.nn import functional as F
    from tqdm import tqdm

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Load everything
    with open(args.data_json) as f: data = json.load(f)
    with open(args.id_to_socket_json) as f: id_to_socket = {int(k): v for k, v in json.load(f).items()}
    with open(args.id_to_node_json) as f: id_to_node = {int(k): v for k, v in json.load(f).items()}

    # Build socket-per-type dictionaries from dataset
    output_sockets = {}
    input_sockets = {}
    for sample in data:
        node_types = sample["node_types"]
        for src, src_sock, dst, dst_sock in sample["edges"]:
            if src < len(node_types) and dst < len(node_types):
                src_type = node_types[src]
                dst_type = node_types[dst]
                output_sockets.setdefault(src_type, set()).add(src_sock)
                input_sockets.setdefault(dst_type, set()).add(dst_sock)

    dataset = EdgeGraphDataset(args.data_json)
    if args.limit_samples:
        dataset = Subset(dataset, range(args.limit_samples))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)

    model = EdgePredictor(args.vocab_size, args.socket_vocab_size, args.d_model,
                          args.nhead, args.nlayers, args.max_nodes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_graphs = 0, 0
        for nodes, lens, edges in tqdm(loader, desc=f"Epoch {epoch+1}"):
            nodes = nodes.to(device)
            edge_logits, exist_logits = model(nodes)

            idx_offset = 0
            loss_list = []
            for b in range(len(edges)):
                n = lens[b].item()
                true_edges = edges[b].to(device)
                pair_count = n * (n - 1)
                if pair_count == 0: continue

                pred_edges = edge_logits[idx_offset:idx_offset+pair_count]
                pred_exist = exist_logits[idx_offset:idx_offset+pair_count]
                idx_offset += pair_count

                labels = torch.full((pair_count, 2), -100, dtype=torch.long, device=device)
                exists = torch.zeros(pair_count, device=device)

                node_ids = nodes[b][:n].tolist()
                node_types = [id_to_node.get(i, None) for i in node_ids]

                for src, src_sock, dst, dst_sock in true_edges.tolist():
                    if src == dst or src >= n or dst >= n:
                        continue

                    src_type = node_types[src]
                    dst_type = node_types[dst]

                    if not src_type or not dst_type:
                        continue

                    valid_src = src_sock in output_sockets.get(src_type, set())
                    valid_dst = dst_sock in input_sockets.get(dst_type, set())

                    if not (valid_src and valid_dst):
                        continue

                    flat_idx = src * (n - 1) + dst - (1 if dst > src else 0)
                    labels[flat_idx] = torch.tensor([src_sock, dst_sock], device=device)
                    exists[flat_idx] = 1.0

                valid = labels[:, 0] != -100
                if valid.sum() == 0: continue

                from_logits = pred_edges[:, :args.socket_vocab_size]
                to_logits = pred_edges[:, args.socket_vocab_size:]

                from_loss = F.cross_entropy(from_logits, labels[:, 0], ignore_index=-100)
                to_loss = F.cross_entropy(to_logits, labels[:, 1], ignore_index=-100)
                exist_loss = F.binary_cross_entropy_with_logits(pred_exist, exists)

                loss = from_loss + to_loss + exist_loss
                loss_list.append(loss)

            if loss_list:
                batch_loss = torch.stack(loss_list).mean()
                opt.zero_grad()
                batch_loss.backward()
                opt.step()
                total_loss += batch_loss.item()
                total_graphs += 1

        print(f"Epoch {epoch+1} avg loss: {total_loss / max(1, total_graphs):.4f}")

    torch.save(model.state_dict(), args.model_out)
    print("Model saved:", args.model_out)

# ─── Sampling ────────────────────────────────────────────────────────────────
# Sample function with filtered fallback socket prediction

def sample(args):
    import json, torch
    from torch.nn.functional import sigmoid
    from collections import defaultdict, Counter

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    with open(args.socket_id_to_name_json) as f: id_to_socket = {int(k): v for k, v in json.load(f).items()}
    with open(args.id_to_node_json) as f: id_to_node = {int(k): v for k, v in json.load(f).items()}
    with open(args.node_type_to_sockets_json) as f: node_type_to_sockets = json.load(f)
    with open("/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_graph_dataset.json") as f: dataset = json.load(f)

    # Build co-occurrence table from dataset
    pair_counts = defaultdict(Counter)
    total_counts = defaultdict(int)
    for sample in dataset:
        node_types = sample["node_types"]
        for src, src_sock, dst, dst_sock in sample["edges"]:
            if src >= len(node_types) or dst >= len(node_types):
                continue
            src_type = node_types[src]
            dst_type = node_types[dst]
            key = (src_sock, dst_type)
            pair_counts[key][dst_sock] += 1
            total_counts[key] += 1

    co_occurrence_probabilities = {}
    for key, counter in pair_counts.items():
        total = total_counts[key]
        sorted_probs = sorted(((dst_sock, count / total) for dst_sock, count in counter.items()), key=lambda x: -x[1])
        co_occurrence_probabilities[key] = sorted_probs

    model = EdgePredictor(args.vocab_size, args.socket_vocab_size, args.d_model,
                          args.nhead, args.nlayers, args.max_nodes).to(device)
    model.load_state_dict(torch.load(args.model_in, map_location=device))
    model.eval()

    node_seq = [int(x) for x in args.node_sequence.strip().split()]
    node_tensor = torch.LongTensor(node_seq).unsqueeze(0).to(device)

    with torch.no_grad():
        edge_logits, exist_logits = model(node_tensor)

    n = len(node_seq)
    from_socks = torch.argmax(edge_logits[:, :args.socket_vocab_size], dim=1)
    to_socks = torch.argmax(edge_logits[:, args.socket_vocab_size:], dim=1)
    probs = sigmoid(exist_logits)

    print("\nPredicted Edges:")
    index = 0
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if probs[index].item() < args.edge_threshold:
                index += 1
                continue

            fs_id = from_socks[index].item()
            ts_id = to_socks[index].item()

            fs = id_to_socket.get(fs_id, "<?>")
            ts = id_to_socket.get(ts_id, "<?>")

            src_type = id_to_node.get(node_seq[i], f"Node{i}")
            dst_type = id_to_node.get(node_seq[j], f"Node{j}")

            valid_from = fs in node_type_to_sockets.get(src_type, {}).get("outputs", [])
            valid_to = ts in node_type_to_sockets.get(dst_type, {}).get("inputs", [])

            # fallback if to_socket is invalid
            if not valid_to:
                fallback = co_occurrence_probabilities.get((fs_id, dst_type))
                if fallback:
                    for fallback_ts_id, _ in fallback:
                        fallback_ts = id_to_socket.get(fallback_ts_id, "<?>")
                        if fallback_ts in node_type_to_sockets.get(dst_type, {}).get("inputs", []):
                            ts_id = fallback_ts_id
                            ts = fallback_ts
                            valid_to = True
                            break

            print(f"{src_type} --[{fs_id}:{fs if valid_from else '<?>'}]--> {dst_type} --[{ts_id}:{ts if valid_to else '<?>'}]")
            index += 1

def validate_edges(dataset_path, id_to_node_path, id_to_socket_path, node_type_to_sockets_path):
    import json
    import argparse
    from pathlib import Path

    # Load JSON files
    with open(dataset_path) as f:
        dataset = json.load(f)

    with open(id_to_node_path) as f:
        id_to_node = {int(k): v for k, v in json.load(f).items()}

    with open(id_to_socket_path) as f:
        id_to_socket = {int(k): v for k, v in json.load(f).items()}

    with open(node_type_to_sockets_path) as f:
        node_type_to_sockets = json.load(f)

    print("Loaded all dictionaries.")
    print(f"Checking {len(dataset)} materials...")

    error_count = 0

    for mat_idx, mat in enumerate(dataset):
        material_name = mat.get("material_name", f"material_{mat_idx}")
        node_types = mat["node_types"]
        edges = mat["edges"]

        for edge in edges:
            from_node_idx, from_socket_id, to_node_idx, to_socket_id = edge

            # Check if indices are in range
            if from_node_idx >= len(node_types) or to_node_idx >= len(node_types):
                print(f"[{material_name}] Edge with invalid node index: {edge}")
                error_count += 1
                continue

            from_type = node_types[from_node_idx]
            to_type = node_types[to_node_idx]

            from_socket_name = id_to_socket.get(from_socket_id, "<??>")
            to_socket_name   = id_to_socket.get(to_socket_id, "<??>")

            from_outputs = node_type_to_sockets.get(from_type, {}).get("outputs", [])
            to_inputs    = node_type_to_sockets.get(to_type, {}).get("inputs", [])

            if from_socket_name not in from_outputs:
                print(f"[{material_name}] {from_type} has no output '{from_socket_name}' (ID {from_socket_id})")
                error_count += 1

            if to_socket_name not in to_inputs:
                print(f"[{material_name}] {to_type} has no input '{to_socket_name}' (ID {to_socket_id})")
                error_count += 1

    print(f"\n Validation complete. Found {error_count} socket issues.")
    if error_count == 0:
        print(" Dataset is clean!")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    p_validate = subparsers.add_parser("validate")
    p_validate.add_argument("--dataset", required=True, help="Path to cleaned dataset JSON")
    p_validate.add_argument("--id-to-node", required=True, help="Path to id_to_node.json")
    p_validate.add_argument("--id-to-socket", required=True, help="Path to id_to_socket.json")
    p_validate.add_argument("--node-type-to-sockets", required=True, help="Path to node_type_to_sockets.json")

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--data-json", required=True)
    p_train.add_argument("--model-out", required=True)
    p_train.add_argument("--id-to-socket-json", required=True)
    p_train.add_argument("--id-to-node-json", required=True)
    p_train.add_argument("--node-type-to-sockets-json", required=True)
    p_train.add_argument("--limit-samples", type=int, default=None,
                         help="Limit the number of training samples for faster debugging")
    p_train.add_argument("--vocab-size", type=int, default=128)
    p_train.add_argument("--socket-vocab-size", type=int, default=64)
    p_train.add_argument("--d-model", type=int, default=256)
    p_train.add_argument("--nhead", type=int, default=4)
    p_train.add_argument("--nlayers", type=int, default=4)
    p_train.add_argument("--max-nodes", type=int, default=64)
    p_train.add_argument("--batch-size", type=int, default=1)
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--lr", type=float, default=1e-4)

    p_sample = subparsers.add_parser("sample")
    p_sample.add_argument("--model-in", required=True)
    p_sample.add_argument("--node-sequence", required=True)
    p_sample.add_argument("--id-to-node-json", required=True)
    p_sample.add_argument("--socket-id-to-name-json", required=True)
    p_sample.add_argument("--node-type-to-sockets-json", required=True)
    p_sample.add_argument("--edge-threshold", type=float, default=0.5)

    # Reuse training params
    p_sample.add_argument("--vocab-size", type=int, default=128)
    p_sample.add_argument("--socket-vocab-size", type=int, default=64)
    p_sample.add_argument("--d-model", type=int, default=256)
    p_sample.add_argument("--nhead", type=int, default=4)
    p_sample.add_argument("--nlayers", type=int, default=4)
    p_sample.add_argument("--max-nodes", type=int, default=64)

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample(args)
    elif args.mode == "validate":
        validate_edges(args.dataset, args.id_to_node, args.id_to_socket, args.node_type_to_sockets)