"""
This script preprocesses a dataset, generates mappings for node types to unique IDs, trains a GPT-2 model, and generates sequences using the trained model. It incorporates additional steps for handling node-to-ID mappings and processing sequences with IDs instead of raw node types.

### Key Components

#### Variables
1. Paths:
   - `database_path`: Path to the raw input JSON database.
   - `cleaned_data_path`: Path to save the preprocessed data (now includes node-to-ID mappings).
   - `model_save_path`: Path to save the trained GPT-2 model and tokenizer.
   - `generated_sequence_path`: Path to save the generated sequences.
   - `node_to_id_path`: Path to save the node-to-ID mapping (JSON file).
   - `id_to_node_path`: Path to save the ID-to-node mapping (JSON file).

2. Device Configuration:
   - `device`: Automatically detects GPU if available; otherwise, defaults to CPU.

#### Functions

1. `generate_node_to_id(input_file, node_to_id_file, id_to_node_file)`:
   - Purpose: Extracts unique node types from the dataset and generates node-to-ID and ID-to-node mappings.
   - Steps:
       - Loads raw JSON data.
       - Extracts unique node types from the dataset.
       - Creates mappings where each node type is assigned a unique integer ID.
       - Saves both mappings to separate JSON files.
   - Returns: `node_to_id` and `id_to_node` dictionaries.

2. `preprocess_dataset_with_ids(input_file, output_file, node_to_id)`:
   - Purpose: Converts node types in the dataset to their corresponding IDs using the node-to-ID mapping.
   - Steps:
       - Loads raw JSON data.
       - Replaces node types with their IDs based on the mapping.
       - Saves preprocessed sequences (as ID strings) to a file.
   - Returns: A list of cleaned sequences (as strings of IDs).

3. `train_model()`:
   - Purpose: Prepares data, trains a GPT-2 model, and saves the trained model. This version includes node-to-ID mapping generation and preprocessing with IDs.
   - Steps:
       - Generates or loads node-to-ID mappings.
       - Preprocesses the dataset by converting node types to IDs.
       - Converts cleaned ID-based data into a custom dataset (`MaterialDataset`).
       - Loads GPT-2 model and tokenizer.
       - Adds a padding token if missing.
       - Trains the model using PyTorch DataLoader with mixed-precision (`torch.amp`).
       - Saves the trained model and tokenizer.
   - Key Components:
       - Node-to-ID Mapping: Added for handling sequences as IDs.
       - `MaterialDataset`: Custom dataset class for ID-based sequences.
       - Mixed-Precision Training: Uses `GradScaler` for efficient GPU utilization.

4. `use_model(start_sequence="46")`:
   - Purpose: Uses the trained model to generate sequences starting from a given ID. Converts generated IDs back to node types for interpretability.
   - Steps:
       - Loads the trained model and tokenizer.
       - Loads the ID-to-node mapping.
       - Tokenizes the input sequence (as IDs).
       - Generates output using GPT-2 with constraints (`num_beams`, `top_k`, `top_p`).
       - Converts generated IDs back to node types using the mapping.
       - Saves the generated sequence (both as IDs and node types) to a file.
   - Arguments:
       - `start_sequence`: Initial ID to start text generation (default: "46").

#### Logging
- Logs all major steps using the `logging` module, including loading data, preprocessing, training progress, and sequence generation.

#### Differences Compared to the Previous Script
1. **Node-to-ID Mapping**:
   - Introduced mappings (`node_to_id` and `id_to_node`) to convert node types to integer IDs.
   - Handles sequences as IDs during preprocessing and training for efficiency and standardization.

2. **Preprocessing**:
   - Replaced node type strings with their corresponding IDs during preprocessing.

3. **Generation Output**:
   - Outputs both ID-based sequences and their corresponding node types for interpretability.

4. **Enhanced Generation Parameters**:
   - Adjusted parameters (`temperature`, `top_k`, `repetition_penalty`, etc.) for more controlled and diverse sequence generation.

#### Example Usage
- Uncomment `train_model()` to train the model (if not already trained).
- Use `use_model()` to generate sequences from the trained model.

### Summary of Variables
| Variable Name             | Description                                                      |
|---------------------------|------------------------------------------------------------------|
| `database_path`           | Path to raw JSON data containing nodes and materials.           |
| `cleaned_data_path`       | Path to save cleaned/preprocessed training data.                |
| `model_save_path`         | Directory to save the trained GPT-2 model and tokenizer.        |
| `generated_sequence_path` | Path to save generated sequences (IDs and node types).          |
| `node_to_id_path`         | Path to save node-to-ID mapping.                                |
| `id_to_node_path`         | Path to save ID-to-node mapping.                                |
| `device`                  | Specifies whether to use GPU (`cuda`) or CPU.                   |

### Summary of Functions
| Function                  | Purpose                                                          |
|---------------------------|------------------------------------------------------------------|
| `generate_node_to_id()`   | Generates mappings between node types and unique IDs.            |
| `preprocess_dataset_with_ids()` | Converts node types to IDs and preprocesses the dataset.       |
| `train_model()`           | Preprocesses data, trains GPT-2, and saves the model.            |
| `use_model()`             | Generates sequences using IDs and converts them back to node types. |
"""


import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.amp import GradScaler, autocast
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
database_path = r"C:\Users\hyperbook\Desktop\PPMGR\Projects\Results\merged_database.json"
cleaned_data_path = r"C:\Users\hyperbook\Desktop\PPMGR\cleaned_training_data.json"
model_save_path = r"C:\Users\hyperbook\Desktop\PPMGR\trained_gpt2_model"
generated_sequence_path = r"C:\Users\hyperbook\Desktop\PPMGR\generated_sequence.txt"
node_to_id_path = r"C:\Users\hyperbook\Desktop\PPMGR\node_to_id.json"
id_to_node_path = r"C:\Users\hyperbook\Desktop\PPMGR\id_to_node.json"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Generate Node-to-ID Mapping
def generate_node_to_id(input_file, node_to_id_file, id_to_node_file):
    logging.info("Loading training data...")
    with open(input_file, "r") as file:
        raw_data = json.load(file)

    logging.info("Extracting unique node types...")
    node_types = set()

    for material_name, nodes in raw_data.get("nodes", {}).items():
        for node in nodes:
            if "type" in node:
                node_types.add(node["type"])

    node_to_id = {node_type: idx + 1 for idx, node_type in enumerate(sorted(node_types))}  # IDs start at 1
    id_to_node = {v: k for k, v in node_to_id.items()}

    with open(node_to_id_file, "w") as file:
        json.dump(node_to_id, file, indent=4)
    with open(id_to_node_file, "w") as file:
        json.dump(id_to_node, file, indent=4)

    logging.info(f"Node-to-ID mapping saved to '{node_to_id_file}'")
    logging.info(f"ID-to-Node mapping saved to '{id_to_node_file}'")

    return node_to_id, id_to_node

# Preprocess Dataset Function with IDs
def preprocess_dataset_with_ids(input_file, output_file, node_to_id):
    logging.info("Loading raw training data...")
    with open(input_file, "r") as file:
        raw_data = json.load(file)

    logging.info("Converting node types to IDs...")
    preprocessed_data = []

    for material_name, nodes in raw_data.get("nodes", {}).items():
        node_ids = [node_to_id[node["type"]] for node in nodes if "type" in node and node["type"] in node_to_id]
        if node_ids:  # Add valid node IDs
            cleaned_sequence = " ".join(map(str, node_ids))
            preprocessed_data.append(cleaned_sequence)

    if not preprocessed_data:
        raise ValueError("No valid sequences found in the raw dataset!")

    with open(output_file, "w") as file:
        json.dump(preprocessed_data, file, indent=4)

    logging.info(f"Preprocessed data with IDs saved to '{output_file}'")
    return preprocessed_data

# Train Model Function
def train_model():
    # Generate or load node-to-ID mapping
    node_to_id, id_to_node = generate_node_to_id(database_path, node_to_id_path, id_to_node_path)

    # Preprocess the dataset
    try:
        training_data = preprocess_dataset_with_ids(database_path, cleaned_data_path, node_to_id)
    except ValueError as e:
        logging.error(e)
        return

    # Ensure the dataset is not empty
    if len(training_data) == 0:
        logging.error("No valid data found for training. Exiting.")
        return

    # Create Dataset and DataLoader
    class MaterialDataset:
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx]

    logging.info("Preparing data for training...")
    dataset = MaterialDataset(training_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    logging.info(f"Dataset size: {len(dataset)}")

    # Load Model and Tokenizer
    logging.info("Loading GPT-2 model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Add padding token if missing
    if tokenizer.pad_token is None:
        logging.info("Adding pad token to tokenizer...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Move the model to GPU
    model.to(device)

    # Train the Model
    logging.info("Starting training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scaler = GradScaler(device='cuda')
    gradient_accumulation_steps = 4

    model.train()
    for i, sequence in enumerate(loader):
        sequence = sequence[0]  # Extract the string from the batch
        tokenized = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True).to(device)

        # Forward pass with mixed precision
        with autocast(device_type='cuda'):
            outputs = model(**tokenized, labels=tokenized["input_ids"])
            loss = outputs.loss / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Perform optimizer step every accumulation step
        if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            logging.info(f"Batch {i+1}/{len(loader)}, Loss: {loss.item():.4f}")

    logging.info("Training completed!")

    # Save the model and tokenizer
    logging.info("Saving trained model and tokenizer...")
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logging.info("Model training and saving completed!")

# Use Model Function
def use_model(start_sequence="46 25 33"):  # Start with IDs
    logging.info("Loading trained model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)

    # Load the ID-to-node mapping
    with open(id_to_node_path, "r") as file:
        id_to_node = json.load(file)

    # Move model to device
    model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(start_sequence, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate sequences
    logging.info("Generating sequence...")
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=25,  # Align with dataset statistics
        num_beams=5,
        do_sample=True,
        temperature=1.3,  # Add more randomness
        top_k=20,         # Consider a broader range of tokens
        top_p=0.9,       # Allow for cumulative probabilities up to 95%
        repetition_penalty=1.5,  # Penalize repetitive tokens more strongly
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id  # Optional: Stop generation early
    )

    # Decode and filter the generated output as IDs
    generated_sequence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    id_list = list(map(int, generated_sequence.split()))

    # Convert IDs back to node types
    node_types = [id_to_node[str(node_id)] for node_id in id_list if str(node_id) in id_to_node]

    logging.info(f"Generated Sequence (IDs): {generated_sequence}")
    logging.info(f"Generated Sequence (Node Types): {' '.join(node_types)}")

    # Save the results
    with open(generated_sequence_path, "w") as file:
        file.write(f"IDs: {generated_sequence}\nNode Types: {' '.join(node_types)}")
    logging.info(f"Filtered generated sequence saved to '{generated_sequence_path}'")

# Example Usage
if __name__ == "__main__":
    # Train the model (only run once or when retraining is required)
    #train_model()

    # Use the trained model to generate a sequence
    use_model()
