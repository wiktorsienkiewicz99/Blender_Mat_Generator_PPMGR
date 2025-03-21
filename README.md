# Blender AI Procedural Material Generator

Project is a toolset for extracting, processing, and analyzing Blender material graphs, with a focus on preparing data for machine learning pipelines. It supports scraping from `.blend` files, flattening node groups, cleaning edge data, generating node/edge dictionaries, and running graph-based neural network models on the resulting data.

## Features

- Extracts full material node graphs from Blender
- Flattens nested groups for clean, interpretable data
- Prepares node/edge JSON files for ML models
- Visualizes graphs for quick inspection
- Provides transformer and GNN-based model templates
- Enables both training and inference on material graphs

## File Breakdown

### Graph Extraction & Processing

| File                         | Description |
|-----------------------------|-------------|
| `scraper_final_automated.py` | Extracts materials, nodes, and edges from `.blend` files into structured JSON |
| `nodes_edges_ids.py`         | Replaces node names with unique IDs and adds node type labels |
| `create_dictionary.py`       | Generates mapping dictionaries for node names and types |
| `edge_databse_preprocess.py` | Cleans and formats edge data across JSON files |
| `plus_preprocess_edge.py`    | Enhances edge data with metadata or extra structure |
| `json_merger.py`             | Merges multiple JSON outputs into unified node/edge files |
| `visualizer.py`              | Visualizes the node-edge graph using `networkx` and `matplotlib` |
| `run_scraper.sh`             | Bash script for batch scraping across a folder of `.blend` files |
| `test_path.py`               | Utility script for validating path logic |

### Machine Learning Models

| File                         | Description |
|-----------------------------|-------------|
| `model_creator.py`           | Builds and trains a model (e.g., GNN or Transformer) on the processed graph data |
| `model_use.py`               | Loads a trained model and runs inference on new graph data |
| `GNN_model.py`               | Implements a Graph Neural Network for processing material graphs |
| `transformer_edges_model.py` | Transformer-based model architecture focused on edge prediction |
| `transformer_node_model.py`  | Transformer model for node-level prediction or classification |

Typical Dependencies:
	•	Python 3.8+
	•	Blender (for data extraction) with materials saved to scene
	•	networkx, matplotlib, numpy
	•	torch, scikit-learn, transformers (if using ML models)

Example Use Case

This toolchain can be used to:
	•	Generate a dataset of Blender materials
	•	Convert them into flat graphs with edge/node features
	•	Train a GNN to predict missing edges
	•	Train a Transformer to classify or embed node types
	•	Visualize results for debugging

PPMGR/
├── blender_data/
│   └── your_files.blend
├── extracted_json/
├── processed/
├── models/
│   └── GNN_model.py
├── scripts/
├── visualizer.py
├── model_creator.py
├── model_use.py
├── ...