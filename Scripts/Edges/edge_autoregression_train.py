'''
python edge_autoregression_train.py \
  --data-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_graph_dataset.json \
  --model-out /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/v2_edge_model.pth \
  --vocab-size 128 \
  --max-nodes 64 \
  --batch-size 8 \
  --epochs 5
  '''


import json
import argparse
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── Dataset ────────────────────────────────────────────────────────────────

class EdgeExistenceDataset(Dataset):
    def __init__(self, json_path: str, max_nodes=50, min_nodes=5):
        with open(json_path, "r") as f:
            raw_samples = json.load(f)

        self.samples = []
        for sample in raw_samples:
            nodes = sample["nodes"]
            node_types = sample["node_types"]
            edges = set((src, dst) for src, _, dst, _ in sample["edges"])

            if not (min_nodes <= len(nodes) <= max_nodes):
                continue

            self.samples.append({
                "node_ids": nodes,
                "node_types": node_types,
                "edge_pairs": edges
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        node_ids = sample["node_ids"]
        node_types = sample["node_types"]
        edge_pairs = sample["edge_pairs"]

        n = len(node_ids)
        node_tensor = torch.LongTensor(node_ids)
        type_tensor = torch.LongTensor([hash(t) % 1000 for t in node_types])

        edge_label = torch.zeros((n, n), dtype=torch.float32)
        for src, dst in edge_pairs:
            if 0 <= src < n and 0 <= dst < n:
                edge_label[src, dst] = 1.0

        return node_tensor, type_tensor, edge_label

def collate_fn(batch):
    node_tensors = [b[0] for b in batch]
    type_tensors = [b[1] for b in batch]
    edge_labels  = [b[2] for b in batch]

    node_lens = torch.LongTensor([len(n) for n in node_tensors])
    padded_nodes = nn.utils.rnn.pad_sequence(node_tensors, batch_first=True, padding_value=0)
    padded_types = nn.utils.rnn.pad_sequence(type_tensors, batch_first=True, padding_value=0)

    return padded_nodes, padded_types, node_lens, edge_labels

# ─── Model ───────────────────────────────────────────────────────────────────

class EdgeExistencePredictor(nn.Module):
    def __init__(self, vocab_size, type_vocab_size, d_model, nhead, nlayers, max_nodes):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.type_emb = nn.Embedding(type_vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_nodes, d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model),
            num_layers=nlayers
        )

        self.edge_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, node_ids, node_types):
        B, T = node_ids.shape
        pos = self.pos_emb(torch.arange(T, device=node_ids.device).unsqueeze(0).expand(B, T))
        h = self.token_emb(node_ids) + self.type_emb(node_types) + pos

        encoded = self.encoder(h.transpose(0, 1)).transpose(0, 1)  # [B, T, D]

        edge_preds = []
        for b in range(B):
            n = (node_ids[b] > 0).sum().item()
            src_idx, dst_idx = torch.meshgrid(
                torch.arange(n, device=node_ids.device),
                torch.arange(n, device=node_ids.device),
                indexing='ij'
            )
            pair_vecs = torch.cat([
                encoded[b, src_idx.flatten()],
                encoded[b, dst_idx.flatten()]
            ], dim=1)
            logits = self.edge_head(pair_vecs).view(n, n)
            edge_preds.append(logits)

        return edge_preds

# ─── Training ────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    dataset = EdgeExistenceDataset(args.data_json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    model = EdgeExistencePredictor(
        vocab_size=args.vocab_size,
        type_vocab_size=1000,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        max_nodes=args.max_nodes
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for node_ids, node_types, lens, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            node_ids = node_ids.to(device)
            node_types = node_types.to(device)
            labels = [l.to(device) for l in labels]

            preds = model(node_ids, node_types)

            loss = 0.0
            for p, t in zip(preds, labels):
                loss += F.binary_cross_entropy_with_logits(p, t)

            loss = loss / len(labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} avg loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), args.model_out)
    print("Model saved to:", args.model_out)

# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train(args)
