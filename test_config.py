import sys
import os

# Add the Scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Scripts"))

# Import the config_loader module
from Scripts.config_loader import load_config

# Try to load the configuration
try:
    config = load_config()
    print("Configuration loaded successfully!")
    print("Configuration:", config)
except Exception as e:
    print("Error loading configuration:", e)