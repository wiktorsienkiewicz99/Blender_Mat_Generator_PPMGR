#!/usr/bin/env python3
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from tqdm import tqdm

# ─── Dataset ────────────────────────────────────────────────────────────────

class EdgeSeqDataset(Dataset):
    """
    Reads a JSON array of strings like:
       "<BOS> 0 12 5 23 6 17 <EOS>"
    where the numbers are pointers into your precomputed slot list.
    """
    def __init__(self, json_path: str, slot_count: int):
        raw = json.load(Path(json_path).open('r'))
        self.slot_count = slot_count
        self.BOS = slot_count       # numeric ID for <BOS>
        self.EOS = slot_count + 1   # numeric ID for <EOS>
        self.PAD = 0                # we will pad with 0

        self.seqs = []
        for line in raw:
            toks = line.strip().split()
            assert toks[0] == "<BOS>" and toks[-1] == "<EOS>"
            body = [int(t) for t in toks[1:-1]]  # slot pointers
            wrapped = [self.BOS] + body + [self.EOS]
            self.seqs.append(torch.LongTensor(wrapped))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]

def collate_edge(batch):
    """
    Pads a batch of 1D LongTensors to [B, T_max] with PAD=0.
    Returns (padded, lengths).
    """
    lengths = torch.LongTensor([t.size(0) for t in batch])
    padded  = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    return padded, lengths

# ─── EdgeGenerator Model ───────────────────────────────────────────────────

class EdgeGenerator(nn.Module):
    def __init__(self, d_model=512, nhead=8, nlayers=6, max_seq_len=512):
        super().__init__()
        # placeholders; will be re‑init in reset_output_size()
        self.token_emb = nn.Embedding(1, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        decoder_layer  = TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer = TransformerDecoder(decoder_layer, nlayers)
        self.out       = nn.Linear(d_model, 1)

    def reset_output_size(self, slot_count: int):
        """
        Rebuild token embeddings & output layer.
        Token IDs range:
          0 = PAD
          1..slot_count = slot pointers
          slot_count+1 = BOS
          slot_count+2 = EOS
        So total tokens = slot_count + 3
        """
        vocab_size = slot_count + 3
        d = self.pos_emb.embedding_dim
        self.token_emb = nn.Embedding(vocab_size, d)
        self.out       = nn.Linear(d, vocab_size)

    def forward(self, tgt_seq: torch.Tensor, slot_memory: torch.Tensor, clip_cond: torch.Tensor = None):
        """
        tgt_seq:    [T, B] token IDs including BOS at row 0
        slot_memory:[S, B, d_model]
        clip_cond:  [B, d_model] optional
        """
        T, B = tgt_seq.shape
        device = tgt_seq.device

        # token + positional embeddings
        tok = self.token_emb(tgt_seq)  # [T,B,d]
        pos_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, B)
        pos     = self.pos_emb(pos_idx)
        h       = tok + pos

        # memory: [1 + S, B, d]
        if clip_cond is not None:
            cond = clip_cond.unsqueeze(0)  # [1,B,d]
        else:
            cond = torch.zeros(1, B, h.size(-1), device=device)
        memory = torch.cat([cond, slot_memory], dim=0)

        # causal mask: upper triangular -inf above diag
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)

        # decode
        dec = self.transformer(tgt=h, memory=memory, tgt_mask=mask)  # [T,B,d]

        # project to vocab
        logits = self.out(dec)  # [T,B,vocab_size]
        return logits

# ─── Training Loop ──────────────────────────────────────────────────────────

def train_edge_generator(
    edge_json_path: str,
    slot_count: int,
    slot_memory_provider,     # fn(idx) -> [S, d_model]
    d_model=512,
    nhead=8,
    nlayers=6,
    batch_size=16,
    epochs=10,
    lr=1e-4,
    device_str="mps"
):
    device = torch.device(device_str if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # dataset + loader
    ds = EdgeSeqDataset(edge_json_path, slot_count)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_edge)

    # model
    model = EdgeGenerator(d_model, nhead, nlayers)
    model.reset_output_size(slot_count)
    model = model.to(device)   # ensures embeddings go to MPS
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD=0

    # training
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for batch_ptrs, lengths in tqdm(loader, desc=f"Epoch {epoch}"):
            B, T = batch_ptrs.shape
            batch_ptrs = batch_ptrs.to(device)  # [B, T]

            # teacher forcing: shift by one
            inp = batch_ptrs[:, :-1]   # [B, T-1]
            tgt = batch_ptrs[:,  1:]   # [B, T-1]
            tgt_in  = inp.transpose(0,1)   # [T-1, B]
            tgt_out = tgt.transpose(0,1)   # [T-1, B]

            # build slot_memory: [S, B, d_model]
            sm = slot_memory_batch(slot_memory_provider, range(B), d_model).to(device)

            # forward & loss
            logits = model(tgt_in, sm)  # [T-1, B, vocab]
            loss   = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch:2d} avg loss: {avg:.4f}")

    return model

# ─── Slot Memory Helper ─────────────────────────────────────────────────────

def slot_memory_batch(provider, batch_indices, d_model):
    """
    provider(i) -> [S, d_model], returns [S, B, d_model]
    """
    slots = [provider(i) for i in batch_indices]
    return torch.stack(slots, dim=1)

# ─── Example Stub & Runner ──────────────────────────────────────────────────

if __name__ == "__main__":
    EDGE_JSON  = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/cleaned_training_data.json"
    SLOT_COUNT = 128  # replace with your actual slot count per graph

    def fake_slot_provider(idx):
        # TODO: replace with your NodeGenerator embeddings → slot embeddings
        return torch.randn(SLOT_COUNT, 512)

    model = train_edge_generator(
        edge_json_path      = EDGE_JSON,
        slot_count          = SLOT_COUNT,
        slot_memory_provider= fake_slot_provider,
        d_model             = 512,
        nhead               = 8,
        nlayers             = 6,
        batch_size          = 8,
        epochs              = 5,
        lr                  = 1e-4,
        device_str          = "mps"
    )

    # save weights
    torch.save(model.state_dict(), "../Edges/edge_generator_mps.pth")