#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer

# ─── EdgeGenerator Definition ───────────────────────────────────────────────

class EdgeGenerator(nn.Module):
    def __init__(self, d_model=512, nhead=8, nlayers=6, max_seq_len=512):
        super().__init__()
        # placeholder; rebuilt in reset_output_size()
        self.token_emb = nn.Embedding(1, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        dec_layer      = TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048)
        self.transformer = TransformerDecoder(dec_layer, nlayers)
        self.out       = nn.Linear(d_model, 1)

    def reset_output_size(self, slot_count: int):
        # tokens: 0 = PAD, 1..slot_count = slots, slot_count+1 = BOS, slot_count+2 = EOS
        vocab_size = slot_count + 3
        d = self.pos_emb.embedding_dim
        self.token_emb = nn.Embedding(vocab_size, d)
        self.out       = nn.Linear(d, vocab_size)

    def forward(self, tgt_seq: torch.Tensor, slot_memory: torch.Tensor, clip_cond: torch.Tensor = None):
        T, B = tgt_seq.shape
        device = tgt_seq.device

        tok = self.token_emb(tgt_seq)  # [T,B,d]
        pos_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, B)
        pos = self.pos_emb(pos_idx)    # [T,B,d]
        h   = tok + pos

        if clip_cond is not None:
            cond = clip_cond.unsqueeze(0)  # [1,B,d]
        else:
            cond = torch.zeros(1, B, h.size(-1), device=device)
        memory = torch.cat([cond, slot_memory], dim=0)  # [1+S, B, d]

        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)

        dec = self.transformer(tgt=h, memory=memory, tgt_mask=mask)  # [T,B,d]
        return self.out(dec)  # [T,B,vocab_size]

# ─── Sampling Function ───────────────────────────────────────────────────────

def generate_edge_sequence(
    model: nn.Module,
    slot_memory: torch.Tensor,
    slot_count: int,
    max_steps: int = 256,
    device: torch.device = torch.device("cpu"),
    top_p: float = 0.9
) -> list[int]:
    model.eval()
    BOS = slot_count + 1
    EOS = slot_count + 2

    seq = torch.tensor([[BOS]], device=device)  # [1,1]
    for _ in range(max_steps):
        logits = model(seq, slot_memory)     # [T,1,V]
        logits = logits[-1, 0]               # [V]
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=0)
        cum   = probs.cumsum(dim=0)
        mask  = cum > top_p
        topk  = (~mask).sum().item() + 1
        topk_idx   = sorted_idx[:topk]
        topk_logits= logits[topk_idx]
        topk_probs = F.softmax(topk_logits, dim=0)
        next_id    = topk_idx[torch.multinomial(topk_probs, 1)].item()

        seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=0)
        if next_id == EOS:
            break

    return seq.squeeze(1).tolist()

# ─── Stub: Slot‑Memory Provider ──────────────────────────────────────────────

def get_slot_memory(graph_index: int) -> torch.Tensor:
    """
    Replace with your NodeGenerator pipeline:
      1) Run node model to get [N, d_model]
      2) Expand each node to its sockets → [S, d_model]
    """
    S = SLOT_COUNT
    d = D_MODEL
    return torch.randn(S, d)

# ─── Main Inference ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Config
    EDGE_MODEL_PATH = "../Edges/edge_generator_mps.pth"
    D_MODEL    = 512
    NHEAD      = 8
    NLAYERS    = 6
    MAX_SEQ    = 512
    SLOT_COUNT = 128  # your actual slot count
    DEVICE     = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load model
    model = EdgeGenerator(d_model=D_MODEL, nhead=NHEAD, nlayers=NLAYERS, max_seq_len=MAX_SEQ)
    model.reset_output_size(SLOT_COUNT)
    state = torch.load(EDGE_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)

    # Sample for some graphs
    for idx in range(3):
        slot_mem = get_slot_memory(idx).unsqueeze(1).to(DEVICE)  # [S,1,d]
        tokens   = generate_edge_sequence(
            model=model,
            slot_memory=slot_mem,
            slot_count=SLOT_COUNT,
            max_steps=256,
            device=DEVICE,
            top_p=0.9
        )
        # strip BOS/EOS and pair safely
        body = tokens[1:-1]
        pairs = []
        for i in range(0, (len(body)//2)*2, 2):
            pairs.append((body[i], body[i+1]))
        if len(body) % 2 == 1:
            print(f"⚠️ Dropping stray final pointer {body[-1]}")
        print(f"\nGraph {idx} → edge slots:", pairs)