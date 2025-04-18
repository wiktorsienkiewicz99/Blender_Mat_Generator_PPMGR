'''python node_autoregression.py train \
  --data-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_training_data.json \
  --id2node-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
  --model-out /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Nodes/node_generator_mps.pth \
  --epochs 5
'''
'''
python node_autoregression.py sample \
  --id2node-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
  --model-in /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Nodes/node_generator_mps.pth \
  --num-samples 5
'''


#!/usr/bin/env python3
import os
# Avoid MPS watermark OOM on inference
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from tqdm import tqdm

# ─── Dataset ────────────────────────────────────────────────────────────────

class NodeSeqDataset(Dataset):
    def __init__(self, data_path: str, id_to_node_path: str):
        id2node = json.load(Path(id_to_node_path).open("r"))
        numeric_ids = sorted(int(k) for k in id2node.keys())
        N = max(numeric_ids)
        # token IDs: 0=PAD, 1..N=node types, N+1=BOS, N+2=EOS
        self.PAD  = 0
        self.BOS  = N + 1
        self.EOS  = N + 2
        self.vocab_size = N + 3

        raw = json.load(Path(data_path).open("r"))
        self.seqs = []
        for line in raw:
            toks = line.strip().split()
            assert toks[0] == "<BOS>" and toks[-1] == "<EOS>"
            ids = []
            for t in toks:
                if t == "<BOS>":  ids.append(self.BOS)
                elif t == "<EOS>":ids.append(self.EOS)
                else:            ids.append(int(t))
            self.seqs.append(torch.LongTensor(ids))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

def collate_nodes(batch):
    lengths = torch.LongTensor([b.size(0) for b in batch])
    padded  = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    return padded, lengths

# ─── Model ───────────────────────────────────────────────────────────────────

class NodeGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        dec_layer      = TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model)
        self.transformer = TransformerDecoder(dec_layer, nlayers)
        self.out       = nn.Linear(d_model, vocab_size)

    def forward(self, seq):
        T, B = seq.shape
        device = seq.device
        tok = self.token_emb(seq)
        pos_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, B)
        pos   = self.pos_emb(pos_idx)
        h     = tok + pos
        mem   = torch.zeros(1, B, h.size(-1), device=device)
        mask  = torch.full((T, T), float("-inf"), device=device)
        mask  = torch.triu(mask, diagonal=1)
        dec   = self.transformer(tgt=h, memory=mem, tgt_mask=mask)
        return self.out(dec)

# ─── Training ────────────────────────────────────────────────────────────────

def train(args):
    # prepare device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Training on", device)

    # dataset & loader
    ds = NodeSeqDataset(args.data_json, args.id2node_json)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_nodes)

    # model
    model = NodeGenerator(
        vocab_size = ds.vocab_size,
        d_model    = args.d_model,
        nhead      = args.nhead,
        nlayers    = args.nlayers,
        max_seq_len= args.max_seq_len
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit= nn.CrossEntropyLoss(ignore_index=ds.PAD)

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        for batch, lengths in tqdm(loader, desc=f"Epoch {epoch}"):
            B, T = batch.shape
            batch = batch.to(device)
            inp = batch[:, :-1].transpose(0,1)
            tgt = batch[:,  1:].transpose(0,1)
            logits = model(inp)
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} avg loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.model_out)
    print("Saved model to", args.model_out)

# ─── Sampling ────────────────────────────────────────────────────────────────

def sample(args):
    # device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Sampling on", device)

    # load id2node to know BOS/EOS, N
    id2node = json.load(Path(args.id2node_json).open("r"))
    numeric_ids = sorted(int(k) for k in id2node.keys())
    N = max(numeric_ids)
    vocab_size = N + 3
    BOS_ID = N + 1
    EOS_ID = N + 2

    # build model & load
    model = NodeGenerator(
        vocab_size = vocab_size,
        d_model    = args.d_model,
        nhead      = args.nhead,
        nlayers    = args.nlayers,
        max_seq_len= args.max_seq_len
    )
    state = torch.load(args.model_in, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    # generate
    for i in range(args.num_samples):
        seq = torch.tensor([[BOS_ID]], device=device)
        with torch.no_grad():
            for _ in range(args.max_len):
                logits = model(seq)[-1,0]
                vals, idxs = torch.sort(logits, descending=True)
                probs = F.softmax(vals, dim=0)
                cum   = probs.cumsum(0)
                k     = (~(cum>args.top_p)).sum().item() + 1
                choices     = idxs[:k]
                choice_probs= F.softmax(vals[:k], dim=0)
                pick = choices[torch.multinomial(choice_probs,1)].item()
                seq = torch.cat([seq, torch.tensor([[pick]], device=device)], dim=0)
                if pick==EOS_ID: break

        tokens = seq.squeeze(1).tolist()
        names = [id2node[str(t)] for t in tokens if 1<=t<=N]
        print(f"\nSample {i+1}: {' → '.join(names)}")

# ─── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-json",    required=True)
    p_train.add_argument("--id2node-json", required=True)
    p_train.add_argument("--model-out",    required=True)
    p_train.add_argument("--epochs",       type=int,   default=3)
    p_train.add_argument("--batch-size",   type=int,   default=32)
    p_train.add_argument("--d-model",      type=int,   default=256)
    p_train.add_argument("--nhead",        type=int,   default=4)
    p_train.add_argument("--nlayers",      type=int,   default=4)
    p_train.add_argument("--max-seq-len",  type=int,   default=256)
    p_train.add_argument("--lr",           type=float, default=1e-4)

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--id2node-json", required=True)
    p_sample.add_argument("--model-in",     required=True)
    p_sample.add_argument("--num-samples", type=int, default=5)
    p_sample.add_argument("--max-len",      type=int, default=64)
    p_sample.add_argument("--top-p",        type=float, default=0.9)
    # reuse training dims so model matches
    p_sample.add_argument("--d-model",      type=int,   default=256)
    p_sample.add_argument("--nhead",        type=int,   default=4)
    p_sample.add_argument("--nlayers",      type=int,   default=4)
    p_sample.add_argument("--max-seq-len",  type=int,   default=256)

    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample(args)