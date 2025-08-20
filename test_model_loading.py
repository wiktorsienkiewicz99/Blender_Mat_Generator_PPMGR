import sys
import os
import torch
from pathlib import Path

# Add the Scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scripts"))

# Import the necessary modules
from Scripts.Nodes.transformer_node_model_upgrade import MaterialAwareGPT
from transformers import AutoModelForCausalLM, AutoConfig

# Path to the model
model_path = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Nodes/material_aware_model.pt"

try:
    print(f"Loading model from {model_path}...")

    # Create a simple model for testing
    config = AutoConfig.from_pretrained("gpt2")
    config.vocab_size = 100  # Just for testing

    # Create the base model
    base_model = AutoModelForCausalLM.from_config(config)

    # Create the material-aware model
    model = MaterialAwareGPT(base_model, material_feature_dim=None)

    # Load the model weights with weights_only=True for security
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    print(f"Loaded state type: {type(state)}")

    if isinstance(state, dict):
        print(f"State keys: {list(state.keys())}")

        if "model_state_dict" in state:
            print("Found model_state_dict in state, loading it...")

            # If material_feature_dim is in the state, use it to initialize the model
            if "material_feature_dim" in state:
                material_feature_dim = state["material_feature_dim"]
                print(f"Found material_feature_dim: {material_feature_dim}")

                # Recreate the model with the correct material_feature_dim and vocab_size
                config.vocab_size = 80  # Based on the error message
                base_model = AutoModelForCausalLM.from_config(config)
                model = MaterialAwareGPT(base_model, material_feature_dim=material_feature_dim)

                # Manually resize the material_adapter and material_projection layers to match the saved dimensions
                import torch.nn as nn
                new_adapter = nn.Linear(material_feature_dim, model.fixed_dim)
                model.material_adapter = new_adapter

                # Also resize the material_projection layer
                # Based on the error message, the projection weight has shape [768, 33470]
                # In PyTorch, nn.Linear(in_features, out_features) creates a weight matrix of shape [out_features, in_features]
                # So we need to create a linear layer with in_features=material_feature_dim, out_features=hidden_size
                hidden_size = base_model.config.hidden_size  # 768 for GPT-2
                new_projection = nn.Linear(material_feature_dim, hidden_size)
                model.material_projection = new_projection

            try:
                # Load the state dict
                model.load_state_dict(state["model_state_dict"])
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading state dict: {e}")
                print("Attempting to load with strict=False...")
                model.load_state_dict(state["model_state_dict"], strict=False)
                print("Model loaded with strict=False!")
        else:
            print("No model_state_dict found in state, trying to load state directly...")
            try:
                model.load_state_dict(state)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading state dict: {e}")
                print("Attempting to load with strict=False...")
                model.load_state_dict(state, strict=False)
                print("Model loaded with strict=False!")
    else:
        print("State is not a dictionary, trying to load it directly...")
        try:
            model.load_state_dict(state)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            print("Attempting to load with strict=False...")
            model.load_state_dict(state, strict=False)
            print("Model loaded with strict=False!")

except Exception as e:
    print(f"Error loading model: {e}")
