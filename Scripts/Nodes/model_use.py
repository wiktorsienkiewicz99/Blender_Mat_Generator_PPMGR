import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Paths
generated_sequence_path = "generated_sequence.txt"
model_path = "./trained_gpt2_model"

# Verify model path exists
if not os.path.exists(model_path):
    logging.error(f"Model path '{model_path}' does not exist!")
    exit()

# Check model directory contents
logging.info(f"Contents of model path: {os.listdir(model_path)}")

# Load the model and tokenizer
logging.info("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Add padding token if missing
if tokenizer.pad_token is None:
    logging.info("Adding pad token to tokenizer...")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.info(f"Using device: {device}")

# Ensure the model is in evaluation mode
model.eval()

# Example starting sequence
start_sequence = "TEX_NOISE FRESNEL MIX_SHADER"

# Tokenize input with attention mask
inputs = tokenizer(start_sequence, return_tensors="pt", padding=True, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate sequences
try:
    logging.info("Generating sequence...")
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
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

except Exception as e:
    logging.error(f"Error during generation: {e}")
