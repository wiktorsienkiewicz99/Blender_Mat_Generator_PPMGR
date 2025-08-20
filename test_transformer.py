import sys
import os

# Add the Scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scripts"))

# Try to import the MaterialAwareGPT class from transformer_node_model_upgrade.py
try:
    from Scripts.Nodes.transformer_node_model_upgrade import MaterialAwareGPT
    print("MaterialAwareGPT class imported successfully!")
    
    # Check if the configuration was loaded correctly
    from Scripts.Nodes.transformer_node_model_upgrade import config
    print("Configuration loaded successfully!")
    print("Configuration keys:", list(config.keys()))
except Exception as e:
    print("Error:", e)