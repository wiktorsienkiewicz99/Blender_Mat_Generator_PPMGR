import os
import platform
import subprocess
import json
import logging

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader, random_split

from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    AutoConfig
)
from config_loader import load_config

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

# -----------------------------------------------------------------------------
# Load configuration & set up paths
# -----------------------------------------------------------------------------
config = load_config()
dataset_refined_folder    = config["dataset_refined_folder_path"]
dataset_auxiliary_folder  = config["dataset_auxiliary_folder_path"]
models_nodes_folder_path  = config["models_nodes_folder_path"]

database_path            = os.path.join(dataset_refined_folder,   "merged_dataset.json")
cleaned_data_path        = os.path.join(dataset_refined_folder,   "cleaned_training_data.json")
tokenized_data_path      = os.path.join(models_nodes_folder_path, "tokenized.pt")
model_save_path          = models_nodes_folder_path
generated_sequence_path  = os.path.join(models_nodes_folder_path, "generated_sequence.txt")
node_to_id_path          = os.path.join(dataset_auxiliary_folder, "node_to_id.json")
id_to_node_path          = os.path.join(dataset_auxiliary_folder, "id_to_node.json")

# -----------------------------------------------------------------------------
# Device detection (CUDA > MPS > CPU)
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info(f"Using device: {device}")

# -----------------------------------------------------------------------------
# Special tokens
# -----------------------------------------------------------------------------
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

# -----------------------------------------------------------------------------
# Build a fixed word-level tokenizer for IDs 1–76 plus specials
# -----------------------------------------------------------------------------
def get_fixed_tokenizer():
    # Build vocab for IDs 1–76 plus specials
    vocab = {str(i): i for i in range(1, 77)}
    vocab.update({PAD_TOKEN: 0, BOS_TOKEN: 77, EOS_TOKEN: 78, UNK_TOKEN: 79})

    tokenizer_obj = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer_obj.pre_tokenizer = Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        unk_token=UNK_TOKEN,
    )
    return tokenizer

# -----------------------------------------------------------------------------
# Preprocess dataset: map node types to IDs, add BOS/EOS, save cleaned json
# -----------------------------------------------------------------------------
def preprocess_dataset_with_ids(input_file, output_file, node_to_id_file):
    logging.info("Loading raw training data...")
    with open(input_file, "r") as f:
        raw_data = json.load(f)
    with open(node_to_id_file, "r") as f:
        node_to_id = json.load(f)

    logging.info("Converting node types to IDs and adding BOS/EOS tokens...")
    sequences = []
    for _, material_data in raw_data.get("materials", {}).items():
        ids = [node_to_id[n.get("type")] for n in material_data.get("nodes", []) if n.get("type") in node_to_id]
        if not ids:
            continue
        seq = [BOS_TOKEN] + [str(i) for i in ids] + [EOS_TOKEN]
        sequences.append(" ".join(seq))

    if not sequences:
        raise ValueError("No valid sequences found in the raw dataset!")

    with open(output_file, "w") as f:
        json.dump(sequences, f, indent=4)
    logging.info(f"Saved {len(sequences)} cleaned sequences to '{output_file}'")
    return sequences

# -----------------------------------------------------------------------------
# Training routine with validation, dropout, and early stopping
# -----------------------------------------------------------------------------
def train_model():
    # Preprocess
    sequences = preprocess_dataset_with_ids(database_path, cleaned_data_path, node_to_id_path)

    # Tokenizer
    tokenizer = get_fixed_tokenizer()

    # Model with higher dropout
    model_config = AutoConfig.from_pretrained(
        "gpt2", attn_pdrop=0.2, resid_pdrop=0.2, embd_pdrop=0.2
    )
    model = AutoModelForCausalLM.from_pretrained("gpt2", config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Tokenize once
    if not os.path.exists(tokenized_data_path):
        tokenized = tokenizer(sequences, padding="longest", truncation=True, return_tensors="pt")
        torch.save(tokenized, tokenized_data_path)

    # Dataset split
    tokenized = torch.load(tokenized_data_path)
    dataset   = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"])
    total     = len(dataset)
    val_size  = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)

    # Optimizer & scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scaler    = GradScaler(enabled=(device.type == "cuda"))
    grad_accum = 4
    max_epochs = 10
    patience   = 2
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, max_epochs+1):
        model.train()
        for step, (input_ids, attention_mask) in enumerate(train_loader, start=1):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            if device.type == "cuda":
                with autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                    loss = outputs.loss / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss / grad_accum
                loss.backward()

            if step % grad_accum == 0:
                if device.type == "cuda":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                logging.info(f"Epoch {epoch} Step {step}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask in val_loader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_ppl = torch.exp(torch.tensor(avg_val_loss))
        logging.info(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, PPL: {val_ppl:.2f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            logging.info(f"Checkpointed epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break

    logging.info("Training finished")

# -----------------------------------------------------------------------------
# Generation routine (unchanged)
# -----------------------------------------------------------------------------
def use_model(start_sequence="1 2", num_candidates=5):
    tokenizer = get_fixed_tokenizer()
    model     = AutoModelForCausalLM.from_pretrained(model_save_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()

    with open(node_to_id_path) as f:
        node_to_id = json.load(f)
    with open(id_to_node_path) as f:
        id_to_node = json.load(f)

    mat_out_id = node_to_id.get("Material Output")
    img_tex_id = node_to_id.get("Image Texture")
    tex_coord_id = node_to_id.get("Texture Coordinate")

    prompt = f"{BOS_TOKEN} {start_sequence} "
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids, attention_mask = enc.input_ids.to(device), enc.attention_mask.to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=80,
        do_sample=True,
        temperature=1.0,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=num_candidates,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    chosen = None
    for seq in outputs:
        ids = seq.tolist()
        if mat_out_id not in ids:
            continue
        if img_tex_id in ids and tex_coord_id not in ids:
            continue
        chosen = ids
        break
    if chosen is None:
        chosen = outputs[0].tolist()

    # Trim and map back to node names
    if tokenizer.eos_token_id in chosen:
        chosen = chosen[:chosen.index(tokenizer.eos_token_id)]
    filtered = [i for i in chosen if i not in {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}]
    node_seq = [id_to_node.get(str(i), "?") for i in filtered]

    with open(generated_sequence_path, "w") as f:
        f.write("IDs:   " + " ".join(map(str, filtered)) + "\n")
        f.write("Nodes: " + " ".join(node_seq) + "\n")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    #train_model()
    use_model()  # uncomment to generate sequences
