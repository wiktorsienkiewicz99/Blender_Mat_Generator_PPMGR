'''
python edge_autoregression.py sample \
  --model-in /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/edge_model.pth \
  --node-sequence "43 9 58 23 58 58 58 41 33 56 29 21 67 17 48 30" \
  --socket-id-to-name-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json \
  --id-to-node-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
  --node-type-to-sockets-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_sockets.json \
  --vocab-size 128 \
  --socket-vocab-size 126
'''
'''
python edge_autoregression.py train \
  --data-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_graph_dataset.json \
  --model-out /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/edge_model.pth \
  --id-to-socket-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json \
  --id-to-node-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
  --node-type-to-sockets-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_sockets.json \
  --vocab-size 128 \
  --socket-vocab-size 126 \
  --limit-samples 2000 \
  --epochs 1
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
        nodes = sample["nodes"]
        edges = sample["edges"]

        # Filter BOS/EOS and keep only numeric tokens
        token_ids = [x for x in nodes if isinstance(x, int)]
        node_tensor = torch.LongTensor(token_ids)
        edge_tensor = torch.LongTensor(edges) if edges else torch.zeros((0, 4), dtype=torch.long)

        return node_tensor, edge_tensor

def collate_graphs(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
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
        self.pos_emb   = nn.Embedding(max_nodes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        self.mlp_edge = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),  # no inplace
            nn.Linear(d_model, 2 * socket_vocab_size)
        )

        self.edge_exists = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),  # no inplace
            nn.Linear(d_model, 1)
        )

    def forward(self, node_seq):
        B, T = node_seq.shape
        device = node_seq.device

        token = self.token_emb(node_seq)  # [B, T, D]
        pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos = self.pos_emb(pos_idx)      # [B, T, D]

        h = token + pos                  # [B, T, D]

        h = h.transpose(0, 1)            # [T, B, D]
        encoded = self.encoder(h).transpose(0, 1)  # [B, T, D]

        node_pairs = []
        for b in range(B):
            vecs = encoded[b]
            n = (node_seq[b] > 0).sum().item()
            src_vecs = vecs[:n].unsqueeze(1).repeat(1, n, 1)
            dst_vecs = vecs[:n].unsqueeze(0).repeat(n, 1, 1)
            pairs = torch.cat([src_vecs, dst_vecs], dim=2).reshape(n * n, -1)
            node_pairs.append(pairs)

        x = torch.cat(node_pairs, dim=0)               # [total_pairs, 2D]
        edge_logits = self.mlp_edge(x)                 # [total_pairs, 2 * socket_vocab]
        exist_logits = self.edge_exists(x).squeeze(1)  # [total_pairs]

        return edge_logits, exist_logits, node_pairs

# ─── Training ────────────────────────────────────────────────────────────────
def train(args):
    import torch
    from torch.utils.data import Subset
    import torch.nn.functional as F
    from tqdm import tqdm
    import json

    torch.autograd.set_detect_anomaly(True)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Training on", device)



    dataset = EdgeGraphDataset(args.data_json)

    # 🚀 Apply sample limit for faster debugging
    if args.limit_samples is not None:
        print(f"⚡ Limiting training to {args.limit_samples} samples")
        dataset = Subset(dataset, list(range(args.limit_samples)))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)

    with open(args.id_to_socket_json, "r") as f:
        id_to_socket = {int(k): v for k, v in json.load(f).items()}
        socket_to_id = {v: k for k, v in id_to_socket.items()}

    with open(args.id_to_node_json, "r") as f:
        id_to_node = {int(k): v for k, v in json.load(f).items()}

    with open(args.node_type_to_sockets_json, "r") as f:
        node_type_to_sockets = json.load(f)

    model = EdgePredictor(
        vocab_size=args.vocab_size,
        socket_vocab_size=args.socket_vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        max_nodes=args.max_nodes
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_graphs = 0

        for batch_nodes, lengths, batch_edges in tqdm(loader, desc=f"Epoch {epoch}"):
            batch_nodes = batch_nodes.to(device)
            edge_logits, exist_logits, _ = model(batch_nodes)

            edge_idx = 0
            loss_accumulator = []
            valid_graphs = 0

            for b in range(len(batch_edges)):
                edges = batch_edges[b]
                n = lengths[b].item()

                preds = edge_logits[edge_idx:]
                exist_pred = exist_logits[edge_idx:]

                expected_pairs = n * n
                available_pairs = preds.shape[0]
                actual_pairs = min(expected_pairs, available_pairs)

                if actual_pairs == 0:
                    continue

                preds = preds[:actual_pairs]
                exist_pred = exist_pred[:actual_pairs]
                edge_idx += actual_pairs

                labels = torch.full((actual_pairs, 2), -100, dtype=torch.long, device=device)
                exist_labels = torch.zeros(actual_pairs, dtype=torch.float32, device=device)

                for edge in edges:
                    src, src_sock, dst, dst_sock = edge.tolist()
                    if src >= n or dst >= n:
                        continue
                    idx = src * n + dst
                    if idx >= actual_pairs:
                        continue
                    labels[idx] = torch.tensor([src_sock, dst_sock], device=device)
                    exist_labels[idx] = 1.0

                mask = labels[:, 0] != -100
                if mask.sum() == 0:
                    continue

                from_logits_raw = preds[:, :args.socket_vocab_size]
                to_logits_raw   = preds[:, args.socket_vocab_size:]

                from_logits = from_logits_raw.clone()
                to_logits = to_logits_raw.clone()

                # 🔒 Apply socket masking per pair
                for idx in range(actual_pairs):
                    src = idx // n
                    dst = idx % n

                    if labels[idx, 0] == -100:
                        continue

                    src_node_id = batch_nodes[b][src].item()
                    dst_node_id = batch_nodes[b][dst].item()
                    src_type = id_to_node.get(src_node_id, None)
                    dst_type = id_to_node.get(dst_node_id, None)

                    if not src_type or not dst_type:
                        continue

                    valid_from_names = node_type_to_sockets.get(src_type, {}).get("outputs", [])
                    valid_to_names   = node_type_to_sockets.get(dst_type, {}).get("inputs", [])

                    valid_from_ids = set(socket_to_id[s] for s in valid_from_names if s in socket_to_id)
                    valid_to_ids   = set(socket_to_id[s] for s in valid_to_names if s in socket_to_id)

                    from_mask = torch.ones(args.socket_vocab_size, device=device) * -1e9
                    to_mask   = torch.ones(args.socket_vocab_size, device=device) * -1e9

                    for fid in valid_from_ids:
                        from_mask[fid] = 0
                    for tid in valid_to_ids:
                        to_mask[tid] = 0

                    from_logits[idx] += from_mask
                    to_logits[idx] += to_mask

                from_loss = F.cross_entropy(from_logits[mask], labels[mask, 0])
                to_loss   = F.cross_entropy(to_logits[mask], labels[mask, 1])
                exist_loss = F.binary_cross_entropy_with_logits(exist_pred, exist_labels)

                total_graph_loss = from_loss + to_loss + exist_loss
                loss_accumulator.append(total_graph_loss)
                valid_graphs += 1

            if valid_graphs > 0:
                batch_total_loss = torch.stack(loss_accumulator).mean()
                opt.zero_grad()
                batch_total_loss.backward()
                opt.step()

                print(f"from_logits[mask]: {from_logits[mask][:3].tolist()}")
                print(f"labels[mask, 0]: {labels[mask, 0][:3].tolist()}")

                print("from_logits shape:", from_logits.shape)
                print("labels shape:", labels.shape)

                total_loss += batch_total_loss.item()
                total_graphs += 1
            print(
                f"[Graph {b}] from_loss={from_loss.item():.2f}, to_loss={to_loss.item():.2f}, exist_loss={exist_loss.item():.2f}")

        print(f"Epoch {epoch} avg loss: {total_loss / max(total_graphs, 1):.4f}")

    torch.save(model.state_dict(), args.model_out)
    print("✅ Model saved to:", args.model_out)

# ─── Sampling ────────────────────────────────────────────────────────────────

def sample(args):
    import torch.nn.functional as F
    import json
    from pathlib import Path

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Sampling on", device)

    # Load mappings
    with open(args.socket_id_to_name_json, "r") as f:
        id_to_socket = {int(k): v for k, v in json.load(f).items()}
    with open(args.id_to_node_json, "r") as f:
        id_to_node = {int(k): v for k, v in json.load(f).items()}
    with open(args.node_type_to_sockets_json, "r") as f:
        node_type_to_sockets = json.load(f)

    # Load model
    model = EdgePredictor(
        vocab_size=args.vocab_size,
        socket_vocab_size=args.socket_vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        max_nodes=args.max_nodes
    ).to(device)

    state = torch.load(args.model_in, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Prepare input
    node_seq = [int(x) for x in args.node_sequence.strip().split()]
    node_tensor = torch.LongTensor(node_seq).unsqueeze(0).to(device)

    with torch.no_grad():
        edge_logits, exist_logits, _ = model(node_tensor)

    from_sockets = torch.argmax(edge_logits[:, :args.socket_vocab_size], dim=1)
    to_sockets   = torch.argmax(edge_logits[:, args.socket_vocab_size:], dim=1)
    edge_probs   = torch.sigmoid(exist_logits)

    n = len(node_seq)
    index = 0
    print("\nPredicted Edges:")
    for i in range(n):
        for j in range(n):
            if i == j:
                index += 1
                continue

            # Filter based on binary edge prediction
            if edge_probs[index].item() < args.edge_threshold:
                index += 1
                continue

            # Socket IDs
            fs_id = from_sockets[index].item()
            ts_id = to_sockets[index].item()
            fs = id_to_socket.get(fs_id, "<?>")
            ts = id_to_socket.get(ts_id, "<?>")

            # Node types
            src_type = id_to_node.get(node_seq[i], f"Node{i}")
            dst_type = id_to_node.get(node_seq[j], f"Node{j}")
            src_outputs = node_type_to_sockets.get(src_type, {}).get("outputs", [])
            dst_inputs  = node_type_to_sockets.get(dst_type, {}).get("inputs", [])

            # Validate socket names
            valid_from = fs in src_outputs
            valid_to   = ts in dst_inputs
            fs_display = fs if valid_from else "<?>"
            ts_display = ts if valid_to else "<?>"

            print(f"{src_type} --[{fs_display}]--> {dst_type} --[{ts_display}]")
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

    print("✅ Loaded all dictionaries.")
    print(f"🔍 Checking {len(dataset)} materials...")

    error_count = 0

    for mat_idx, mat in enumerate(dataset):
        material_name = mat.get("material_name", f"material_{mat_idx}")
        node_types = mat["node_types"]
        edges = mat["edges"]

        for edge in edges:
            from_node_idx, from_socket_id, to_node_idx, to_socket_id = edge

            # Check if indices are in range
            if from_node_idx >= len(node_types) or to_node_idx >= len(node_types):
                print(f"❌ [{material_name}] Edge with invalid node index: {edge}")
                error_count += 1
                continue

            from_type = node_types[from_node_idx]
            to_type = node_types[to_node_idx]

            from_socket_name = id_to_socket.get(from_socket_id, "<??>")
            to_socket_name   = id_to_socket.get(to_socket_id, "<??>")

            from_outputs = node_type_to_sockets.get(from_type, {}).get("outputs", [])
            to_inputs    = node_type_to_sockets.get(to_type, {}).get("inputs", [])

            if from_socket_name not in from_outputs:
                print(f"⚠️  [{material_name}] {from_type} has no output '{from_socket_name}' (ID {from_socket_id})")
                error_count += 1

            if to_socket_name not in to_inputs:
                print(f"⚠️  [{material_name}] {to_type} has no input '{to_socket_name}' (ID {to_socket_id})")
                error_count += 1

    print(f"\n🔎 Validation complete. Found {error_count} socket issues.")
    if error_count == 0:
        print("🎉 Dataset is clean!")

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