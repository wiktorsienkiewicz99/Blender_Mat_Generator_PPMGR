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

    logging.info("Converting node types to IDs and adding BOS/EOS tokens with material name context...")
    sequences = []
    material_names = []

    for material_name, material_data in raw_data.get("materials", {}).items():
        ids = [node_to_id[n.get("type")] for n in material_data.get("nodes", []) if n.get("type") in node_to_id]
        if not ids:
            continue
        seq = [BOS_TOKEN] + [str(i) for i in ids] + [EOS_TOKEN]
        sequences.append(" ".join(seq))
        material_names.append(material_name)

    if not sequences:
        raise ValueError("No valid sequences found in the raw dataset!")

    # Create a dictionary with material names and sequences
    data_with_context = {
        "sequences": sequences,
        "material_names": material_names
    }

    with open(output_file, "w") as f:
        json.dump(data_with_context, f, indent=4)
    logging.info(f"Saved {len(sequences)} cleaned sequences with material names to '{output_file}'")
    return sequences, material_names

# -----------------------------------------------------------------------------
# Training routine with validation, dropout, and early stopping
# -----------------------------------------------------------------------------
def train_model():
    # Preprocess
    sequences, material_names = preprocess_dataset_with_ids(database_path, cleaned_data_path, node_to_id_path)

    # Tokenizer
    tokenizer = get_fixed_tokenizer()

    # Model with higher dropout
    model_config = AutoConfig.from_pretrained(
        "gpt2", attn_pdrop=0.2, resid_pdrop=0.2, embd_pdrop=0.2
    )
    model = AutoModelForCausalLM.from_pretrained("gpt2", config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Process material names - convert to embeddings
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    # Create TF-IDF vectorizer for material names
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5))
    material_name_features = vectorizer.fit_transform(material_names)
    material_name_features = normalize(material_name_features, norm='l2')
    material_name_features = torch.tensor(material_name_features.toarray(), dtype=torch.float32)

    # Save vectorizer for later use during generation
    import pickle
    with open(os.path.join(models_nodes_folder_path, "material_name_vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)

    # Tokenize once
    if not os.path.exists(tokenized_data_path):
        tokenized = tokenizer(sequences, padding="longest", truncation=True, return_tensors="pt")
        torch.save({
            "tokenized": tokenized,
            "material_name_features": material_name_features
        }, tokenized_data_path)

    # Dataset split
    data = torch.load(tokenized_data_path)
    tokenized = data["tokenized"] if "tokenized" in data else data
    material_name_features = data.get("material_name_features", material_name_features)

    dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"], material_name_features)
    total = len(dataset)
    val_size = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Create a custom model that incorporates material name features
    class MaterialAwareGPT(nn.Module):
        def __init__(self, base_model, material_feature_dim):
            super().__init__()
            self.base_model = base_model
            # Use a fixed intermediate dimension for material features
            self.fixed_dim = 128
            self.material_adapter = nn.Linear(1, self.fixed_dim)  # Will be resized dynamically
            self.material_projection = nn.Linear(self.fixed_dim, self.base_model.config.hidden_size)
            self.material_attention = nn.Linear(self.base_model.config.hidden_size, 1)

        def forward(self, input_ids, attention_mask, material_features, labels=None):
            # Ensure material_features has at least 2 dimensions
            if material_features.dim() == 1:
                material_features = material_features.unsqueeze(0)

            # Resize the adapter layer if needed
            input_dim = material_features.size(1)
            if self.material_adapter.in_features != input_dim:
                # Create a new adapter layer with the correct input dimension
                new_adapter = nn.Linear(input_dim, self.fixed_dim).to(material_features.device)
                # Initialize with xavier uniform weights
                nn.init.xavier_uniform_(new_adapter.weight)
                self.material_adapter = new_adapter

            # Apply the adapter followed by the projection
            adapted_features = self.material_adapter(material_features)
            material_embedding = self.material_projection(adapted_features).unsqueeze(1)  # [batch, 1, hidden_size]

            # Get base model outputs
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

            # Get the hidden states
            hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

            # Apply material context through attention
            attention_scores = self.material_attention(hidden_states).squeeze(-1)  # [batch, seq_len]
            attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]

            # Apply attention weights to hidden states
            weighted_hidden = hidden_states * attention_weights

            # Add material context
            material_context = material_embedding.expand(-1, hidden_states.size(1), -1)
            enhanced_hidden = weighted_hidden + material_context

            # Replace the last hidden state in the outputs
            outputs.hidden_states = outputs.hidden_states[:-1] + (enhanced_hidden,)

            return outputs

    # Wrap the base model with our custom model
    material_feature_dim = material_name_features.size(1)
    material_aware_model = MaterialAwareGPT(model, material_feature_dim).to(device)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)

    # Optimizer & scaler
    optimizer = torch.optim.AdamW(material_aware_model.parameters(), lr=1e-5, weight_decay=0.01)
    scaler    = GradScaler(enabled=(device.type == "cuda"))
    grad_accum = 4
    max_epochs = 1
    patience   = 2
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, max_epochs+1):
        material_aware_model.train()
        for step, (input_ids, attention_mask, material_features) in enumerate(train_loader, start=1):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            material_features = material_features.to(device)

            if device.type == "cuda":
                with autocast():
                    outputs = material_aware_model(input_ids=input_ids, attention_mask=attention_mask, 
                                                  material_features=material_features, labels=input_ids)
                    loss = outputs.loss / grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = material_aware_model(input_ids=input_ids, attention_mask=attention_mask, 
                                              material_features=material_features, labels=input_ids)
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
        material_aware_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, material_features in val_loader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                material_features = material_features.to(device)
                outputs = material_aware_model(input_ids=input_ids, attention_mask=attention_mask, 
                                              material_features=material_features, labels=input_ids)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_ppl = torch.exp(torch.tensor(avg_val_loss))
        logging.info(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, PPL: {val_ppl:.2f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save the base model
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)

            # Save the material-aware model
            material_model_path = os.path.join(model_save_path, "node_model.pt")
            torch.save({
                "model_state_dict": material_aware_model.state_dict(),
                "material_feature_dim": material_feature_dim,
                "base_model_path": model_save_path
            }, material_model_path)

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
def use_model(start_sequence="1 2", num_candidates=5, material_name=None):
    tokenizer = get_fixed_tokenizer()
    base_model = AutoModelForCausalLM.from_pretrained(model_save_path)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.to(device)

    # Load the material-aware model if material_name is provided
    material_aware_model = None
    if material_name:
        material_model_path = os.path.join(model_save_path, "material_aware_model.pt")
        if os.path.exists(material_model_path):
            # Load the vectorizer
            import pickle
            vectorizer_path = os.path.join(models_nodes_folder_path, "material_name_vectorizer.pkl")
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)

            # Process the material name
            material_features = vectorizer.transform([material_name])
            material_features = torch.tensor(material_features.toarray(), dtype=torch.float32).to(device)

            # Load the material-aware model
            checkpoint = torch.load(material_model_path, map_location=device)
            material_feature_dim = checkpoint["material_feature_dim"]

            # Recreate the MaterialAwareGPT class
            class MaterialAwareGPT(nn.Module):
                def __init__(self, base_model, material_feature_dim):
                    super().__init__()
                    self.base_model = base_model
                    # Use a fixed intermediate dimension for material features
                    self.fixed_dim = 128
                    self.material_adapter = nn.Linear(1, self.fixed_dim)  # Will be resized dynamically
                    self.material_projection = nn.Linear(self.fixed_dim, self.base_model.config.hidden_size)
                    self.material_attention = nn.Linear(self.base_model.config.hidden_size, 1)

                def forward(self, input_ids, attention_mask, material_features, labels=None):
                    # Ensure material_features has at least 2 dimensions
                    if material_features.dim() == 1:
                        material_features = material_features.unsqueeze(0)

                    # Resize the adapter layer if needed
                    input_dim = material_features.size(1)
                    if self.material_adapter.in_features != input_dim:
                        # Create a new adapter layer with the correct input dimension
                        new_adapter = nn.Linear(input_dim, self.fixed_dim).to(material_features.device)
                        # Initialize with xavier uniform weights
                        nn.init.xavier_uniform_(new_adapter.weight)
                        self.material_adapter = new_adapter

                    # Apply the adapter followed by the projection
                    adapted_features = self.material_adapter(material_features)
                    material_embedding = self.material_projection(adapted_features).unsqueeze(1)  # [batch, 1, hidden_size]

                    # Get base model outputs
                    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)

                    # Get the hidden states
                    hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

                    # Apply material context through attention
                    attention_scores = self.material_attention(hidden_states).squeeze(-1)  # [batch, seq_len]
                    attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]

                    # Apply attention weights to hidden states
                    weighted_hidden = hidden_states * attention_weights

                    # Add material context
                    material_context = material_embedding.expand(-1, hidden_states.size(1), -1)
                    enhanced_hidden = weighted_hidden + material_context

                    # Replace the last hidden state in the outputs
                    outputs.hidden_states = outputs.hidden_states[:-1] + (enhanced_hidden,)

                    return outputs

                def generate(self, input_ids, attention_mask, material_features, **kwargs):
                    # For generation, we'll use the base model but enhance its hidden states with material context
                    # This is a simplified approach - in a real implementation, you might want to modify the generation process
                    # to incorporate material context at each step

                    # Ensure material_features has at least 2 dimensions
                    if material_features.dim() == 1:
                        material_features = material_features.unsqueeze(0)

                    # Resize the adapter layer if needed
                    input_dim = material_features.size(1)
                    if self.material_adapter.in_features != input_dim:
                        # Create a new adapter layer with the correct input dimension
                        new_adapter = nn.Linear(input_dim, self.fixed_dim).to(material_features.device)
                        # Initialize with xavier uniform weights
                        nn.init.xavier_uniform_(new_adapter.weight)
                        self.material_adapter = new_adapter

                    # Apply the adapter followed by the projection
                    adapted_features = self.material_adapter(material_features)
                    material_embedding = self.material_projection(adapted_features).unsqueeze(1)

                    # Use the base model for generation
                    return self.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

            # Create and load the material-aware model
            material_aware_model = MaterialAwareGPT(base_model, material_feature_dim).to(device)
            material_aware_model.load_state_dict(checkpoint["model_state_dict"])
            material_aware_model.eval()

            logging.info(f"Using material-aware model with material name: {material_name}")
        else:
            logging.warning(f"Material-aware model not found at {material_model_path}, using base model instead")

    # Use the appropriate model
    model = material_aware_model if material_aware_model else base_model
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

    # Generate with the appropriate model
    if material_aware_model and material_name:
        outputs = material_aware_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            material_features=material_features,
            max_length=80,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=num_candidates,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    else:
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
    train_model()
    #use_model()  # uncomment to generate sequences
