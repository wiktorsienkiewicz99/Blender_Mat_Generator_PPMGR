import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.amp import GradScaler, autocast
import logging
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config_loader import load_config

config = load_config()

# Paths and config
dataset_refined_folder = config["dataset_refined_folder_path"]
dataset_auxiliary_folder = config["dataset_auxiliary_folder_path"]
models_nodes_folder_path = config["models_nodes_folder_path"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
database_path = dataset_refined_folder + "/merged_dataset.json"
cleaned_data_path = dataset_refined_folder + "/cleaned_training_data.json"
model_save_path = models_nodes_folder_path
generated_sequence_path = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/test_dump/saved_sequence.txt"
node_to_id_path = dataset_auxiliary_folder + "/node_to_id.json"
id_to_node_path = dataset_auxiliary_folder + "/id_to_node.json"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

def generate_node_to_id(input_file, node_to_id_file, id_to_node_file):
    with open(input_file, "r") as file:
        raw_data = json.load(file)

    node_types = set()
    for material_name, material_data in raw_data.get("materials", {}).items():
        for node in material_data.get("nodes", []):
            if "type" in node:
                node_types.add(node["type"])

    node_to_id = {node_type: idx + 1 for idx, node_type in enumerate(sorted(node_types))}
    id_to_node = {v: k for k, v in node_to_id.items()}

    with open(node_to_id_file, "w") as file:
        json.dump(node_to_id, file, indent=4)
    with open(id_to_node_file, "w") as file:
        json.dump(id_to_node, file, indent=4)

    return node_to_id, id_to_node

def preprocess_dataset_with_ids(input_file, output_file, node_to_id_file, test_ratio=0.2):
    with open(input_file, "r") as file:
        raw_data = json.load(file)
    with open(node_to_id_file, "r") as file:
        node_to_id = json.load(file)

    preprocessed_data = []
    for material_name, material_data in raw_data.get("materials", {}).items():
        nodes = material_data.get("nodes", [])
        node_ids = [node_to_id[node["type"]] for node in nodes if "type" in node and node["type"] in node_to_id]
        if node_ids:
            cleaned_sequence = " ".join(map(str, node_ids))
            preprocessed_data.append(cleaned_sequence)

    if not preprocessed_data:
        raise ValueError("No valid sequences found in the raw dataset!")

    with open(output_file, "w") as file:
        json.dump(preprocessed_data, file, indent=4)

    return train_test_split(preprocessed_data, test_size=test_ratio, random_state=42)

def train_model():
    train_data, test_data = preprocess_dataset_with_ids(database_path, cleaned_data_path, node_to_id_path)

    if len(train_data) == 0 or len(test_data) == 0:
        logging.error("Empty training or test set!")
        return

    class MaterialDataset:
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx]

    batch_size = 8
    epochs = 3
    learning_rate = 2e-5
    gradient_accumulation_steps = 2

    dataset = MaterialDataset(train_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=epochs * len(loader))
    scaler = GradScaler(device='cuda')

    training_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(loader):
            tokenized = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True).to(device)

            with autocast(device_type='cuda'):
                outputs = model(**tokenized, labels=tokenized["input_ids"])
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item()

            if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()

        avg_train_loss = total_loss / len(loader)
        training_losses.append(avg_train_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for sequence in test_data:
                tokenized = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True).to(device)
                outputs = model(**tokenized, labels=tokenized["input_ids"])
                total_eval_loss += outputs.loss.item()

        avg_test_loss = total_eval_loss / len(test_data)
        test_losses.append(avg_test_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Test Loss: {avg_test_loss:.4f}")

    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Plot loss curves
    plt.plot(training_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, "loss_plot.png"))
    plt.close()

def use_model(start_sequence="21 22"):
    model = AutoModelForCausalLM.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)

    with open(id_to_node_path, "r") as file:
        id_to_node = json.load(file)

    model.to(device)
    model.eval()

    inputs = tokenizer(start_sequence, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=32,
        num_beams=5,
        do_sample=True,
        temperature=1.0,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_sequence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    id_list = list(map(int, generated_sequence.split()))
    node_types = [id_to_node[str(node_id)] for node_id in id_list if str(node_id) in id_to_node]

    with open(generated_sequence_path, "w") as file:
        file.write(f"IDs: {generated_sequence}\nNode Types: {' '.join(node_types)}")

    logging.info(f"Generated Sequence (IDs): {generated_sequence}")
    logging.info(f"Generated Sequence (Node Types): {' '.join(node_types)}")

if __name__ == "__main__":
    #train_model()
    use_model()
