"""
This script preprocesses a dataset, trains a GPT-2 model, and generates sequences using the trained model.
It uses the Hugging Face `transformers` library for GPT-2 and PyTorch for handling datasets and training.

### Key Components

#### Variables
1. Paths:
   - `database_path`: Path to the raw input JSON database.
   - `cleaned_data_path`: Path to save the preprocessed data.
   - `model_save_path`: Path to save the trained GPT-2 model and tokenizer.
   - `generated_sequence_path`: Path to save the generated sequences.

2. Device Configuration:
   - `device`: Automatically detects GPU if available; otherwise, defaults to CPU.

#### Functions

1. preprocess_dataset(input_file, output_file):
   - Purpose: Cleans and extracts meaningful sequences from raw JSON data.
   - Inputs: 
       - `input_file`: Path to the raw data JSON.
       - `output_file`: Path to save the cleaned data.
   - Process:
       - Loads raw JSON data.
       - Extracts unique node types from materials.
       - Saves the cleaned data to a file.
   - Returns: A list of cleaned sequences.

2. train_model():
   - Purpose: Prepares data, trains a GPT-2 model, and saves the trained model.
   - Steps:
       - Preprocesses the dataset.
       - Converts cleaned data into a custom dataset (`MaterialDataset`).
       - Loads GPT-2 model and tokenizer.
       - Adds a padding token if missing.
       - Trains the model using PyTorch DataLoader with mixed-precision (`torch.amp`).
       - Saves the trained model and tokenizer.
   - Key Components:
       - `MaterialDataset`: Custom dataset class for sequences.
       - Mixed-Precision Training: Uses `GradScaler` for efficient GPU utilization.

3. use_model(start_sequence="MAPPING"):
   - Purpose: Uses the trained model to generate sequences starting from a given token.
   - Steps:
       - Loads the trained model and tokenizer.
       - Tokenizes the input sequence.
       - Generates output using GPT-2 with constraints (`num_beams`, `top_k`, `top_p`).
       - Removes duplicate tokens from the output.
       - Saves the cleaned sequence to a file.
   - Arguments:
       - `start_sequence`: Initial token for text generation (default: "MAPPING").

#### Logging
- Logs all major steps using the `logging` module, including loading data, preprocessing, training progress, and sequence generation.

#### Example Usage
- Uncomment `train_model()` to train the model (if not already trained).
- Use `use_model()` to generate sequences from the trained model.

### Summary of Variables
| Variable Name             | Description                                                      |
|---------------------------|------------------------------------------------------------------|
| `database_path`           | Path to raw JSON data containing nodes and materials.           |
| `cleaned_data_path`       | Path to save cleaned/preprocessed training data.                |
| `model_save_path`         | Directory to save the trained GPT-2 model and tokenizer.        |
| `generated_sequence_path` | Path to save filtered generated sequences.                      |
| `device`                  | Specifies whether to use GPU (`cuda`) or CPU (`cpu`).           |

### Summary of Functions
| Function                  | Purpose                                                          |
|---------------------------|------------------------------------------------------------------|
| `preprocess_dataset()`    | Cleans the raw JSON data and extracts training sequences.        |
| `train_model()`           | Preprocesses data, trains GPT-2, and saves the model.            |
| `use_model()`             | Generates sequences using the trained GPT-2 model.              |
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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Preprocess Dataset Function
def preprocess_dataset(input_file, output_file):
    logging.info("Loading raw training data...")
    with open(input_file, "r") as file:
        raw_data = json.load(file)

    logging.info("Cleaning and preprocessing training data...")
    preprocessed_data = []

    # Iterate through each material
    for material_name, nodes in raw_data.get("nodes", {}).items():
        # Extract unique node types
        node_types = list(dict.fromkeys(node["type"] for node in nodes if "type" in node))
        if node_types:  # Only add if there are valid node types
            cleaned_sequence = " ".join(node_types)
            preprocessed_data.append(cleaned_sequence)

    # Ensure the dataset is not empty
    if not preprocessed_data:
        raise ValueError("No valid sequences found in the raw dataset!")

    logging.info("Saving preprocessed data...")
    with open(output_file, "w") as file:
        json.dump(preprocessed_data, file, indent=4)

    logging.info(f"Preprocessed data saved to '{output_file}'")
    return preprocessed_data

# Train Model Function
def train_model():
    # Preprocess the dataset
    try:
        training_data = preprocess_dataset(database_path, cleaned_data_path)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
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
def use_model(start_sequence="MAPPING"):
    logging.info("Loading trained model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)

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
        max_length=300,
        num_beams=5,
        do_sample=True,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )

    # Decode and filter the generated output
    generated_sequence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    unique_tokens = " ".join(dict.fromkeys(generated_sequence.split()))  # Remove duplicates

    logging.info(f"Generated Sequence: {unique_tokens}")

    # Save the filtered generated sequence
    with open(generated_sequence_path, "w") as file:
        file.write(unique_tokens)
    logging.info(f"Filtered generated sequence saved to '{generated_sequence_path}'")

# Example Usage
if __name__ == "__main__":
    # Train the model (only run once or when retraining is required)
    #train_model()

    # Use the trained model to generate a sequence
    use_model()
