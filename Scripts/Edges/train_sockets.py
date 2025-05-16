'''
python train_sockets.py \
  --data-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_graph_dataset.json \
  --model-out /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/sockets_model.pth \
  --d-model 128 \
  --batch-size 32 \
  --epochs 1000
'''

import json
import argparse
from tqdm import tqdm
from typing import List
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── Dataset ────────────────────────────────────────────────────────────────

class SocketAssignmentDataset(Dataset):
    def __init__(self, json_path: str, max_nodes=50, min_nodes=5):
        with open(json_path, "r") as f:
            raw_samples = json.load(f)

        self.samples = []
        for sample in raw_samples:
            if not (min_nodes <= len(sample["nodes"]) <= max_nodes):
                continue

            node_types = sample["node_types"]
            for src, src_sock, dst, dst_sock in sample["edges"]:
                if src >= len(node_types) or dst >= len(node_types):
                    continue
                src_type = node_types[src]
                dst_type = node_types[dst]
                self.samples.append((src_type, dst_type, src_sock, dst_sock))

        # Build vocabularies for node types and sockets
        self.type_to_idx = {t: i for i, t in enumerate(set(t for t, _, _, _ in self.samples) | set(t for _, t, _, _ in self.samples))}
        self.sock_to_idx = {s: i for i, s in enumerate(set(s for _, _, s, _ in self.samples) | set(s for _, _, _, s in self.samples))}
        self.idx_to_sock = {i: s for s, i in self.sock_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_type, dst_type, src_sock, dst_sock = self.samples[idx]
        src_type_idx = self.type_to_idx[src_type]
        dst_type_idx = self.type_to_idx[dst_type]
        src_sock_idx = self.sock_to_idx[src_sock]
        dst_sock_idx = self.sock_to_idx[dst_sock]

        return torch.LongTensor([src_type_idx, dst_type_idx]), torch.LongTensor([src_sock_idx, dst_sock_idx])

# ─── Model ───────────────────────────────────────────────────────────────────

class SocketClassifier(nn.Module):
    def __init__(self, type_vocab_size, sock_vocab_size, d_model):
        super().__init__()
        self.src_type_emb = nn.Embedding(type_vocab_size, d_model)
        self.dst_type_emb = nn.Embedding(type_vocab_size, d_model)

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2 * sock_vocab_size)
        )

    def forward(self, type_pairs):
        src = self.src_type_emb(type_pairs[:, 0])
        dst = self.dst_type_emb(type_pairs[:, 1])
        combined = torch.cat([src, dst], dim=1)
        return self.mlp(combined)

# ─── Training ────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    dataset = SocketAssignmentDataset(args.data_json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SocketClassifier(
        type_vocab_size=len(dataset.type_to_idx),
        sock_vocab_size=len(dataset.sock_to_idx),
        d_model=args.d_model
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for type_pairs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            type_pairs = type_pairs.to(device)
            labels = labels.to(device)

            logits = model(type_pairs)
            pred_src, pred_dst = logits[:, :len(dataset.sock_to_idx)], logits[:, len(dataset.sock_to_idx):]

            loss_src = F.cross_entropy(pred_src, labels[:, 0])
            loss_dst = F.cross_entropy(pred_dst, labels[:, 1])
            loss = loss_src + loss_dst

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} avg loss: {total_loss / len(loader):.4f}")

    torch.save({
        "model": model.state_dict(),
        "type_to_idx": dataset.type_to_idx,
        "sock_to_idx": dataset.sock_to_idx,
        "idx_to_sock": dataset.idx_to_sock
    }, args.model_out)
    print("Model saved to:", args.model_out)

# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    train(args)
