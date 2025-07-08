# Blender Material Generator - Project Summary

## Project Overview
This project is an AI-powered Blender Material Generator that uses machine learning models to automatically generate materials for Blender. The system follows a pipeline approach:

1. **Node Generation**: Predicts a sequence of nodes using transformer-based autoregression
2. **Edge Prediction**: Predicts connections between nodes using Graph Neural Networks (GNN)
3. **Parameter Prediction**: Predicts parameters for nodes using multi-head predictors
4. **Texture Generation**: Generates textures for IMAGE TEX nodes using Stable Diffusion
5. **Blender Integration**: Imports the generated material graph into Blender

## Code Statistics
- **Total Python Files**: 58
- **Total Lines of Code**: 7,271
- **Main Workflow File**: Scripts/material_generation_workflow.py

## Project Structure
```
.
├── Dataset/                  # Contains training and auxiliary data
│   ├── Auxiliary/            # Supporting data files
│   ├── Generated/            # Output from the generation process
│   ├── Raw/                  # Raw data
│   └── Refined/              # Processed data
├── Models/                   # Trained models
│   ├── Edges/                # Edge prediction models
│   ├── Nodes/                # Node generation models
│   └── Parameters/           # Parameter prediction models
├── Projects/                 # Example projects
│   └── assets/               # Project assets
├── Scripts/                  # Main code
│   ├── Blender/              # Blender integration scripts
│   ├── Dataset/              # Dataset processing scripts
│   ├── Edges/                # Edge prediction scripts
│   ├── Nodes/                # Node generation scripts
│   ├── Parameters/           # Parameter prediction scripts
│   └── Textures/             # Texture generation scripts
└── assets/                   # Generated materials and textures
```

## Key Components

### 1. Node Generation
- **Key Files**: 
  - Scripts/Nodes/node_autoregression.py
  - Scripts/Nodes/transformer_node_model.py
  - Scripts/Nodes/GNN_model.py
- **Models Used**: Transformer-based models for autoregressive node generation
- **Functionality**: Generates a sequence of Blender material nodes

### 2. Edge Prediction
- **Key Files**:
  - Scripts/Edges/gnn_edge_predictor.py
  - Scripts/Edges/GNN_edge_model.py
  - Scripts/Edges/gnn_edge_sampler.py
- **Models Used**: Graph Neural Networks (GNN) for predicting connections between nodes
- **Functionality**: Predicts which nodes should be connected and how

### 3. Parameter Prediction
- **Key Files**:
  - Scripts/Parameters/gnn_edge_and_param_predictor.py
  - Scripts/Parameters/test_param_predictor.py
- **Models Used**: Multi-head predictors for parameter prediction
- **Functionality**: Predicts parameters for each node in the material graph

### 4. Texture Generation
- **Key Files**:
  - Scripts/Textures/texture_generator.py
  - Scripts/Textures/SD_CLIP_guided_texture_generator.py
  - Scripts/Textures/SD_prompt_2_texture_generator.py
- **Models Used**: Stable Diffusion for texture generation
- **Functionality**: Generates textures for IMAGE TEX nodes based on prompts

### 5. Blender Integration
- **Key Files**:
  - Scripts/Blender/import_predicted_material.py
- **Functionality**: Imports the generated material graph into Blender

## Dependencies
The project relies on several key libraries:
- **PyTorch** and **torchvision** for deep learning
- **torch-geometric** for graph neural networks
- **transformers** for transformer models
- **diffusers** for Stable Diffusion
- **scikit-learn** for machine learning
- **networkx** for graph operations
- **tqdm** for progress bars
- **pillow** for image processing

## Workflow
The main workflow is orchestrated by the `material_generation_workflow.py` script, which:
1. Generates node sequences using the node generator
2. Predicts edges between nodes using the edge predictor
3. Predicts parameters for nodes using the parameter predictor
4. Optionally generates textures for IMAGE TEX nodes
5. Imports the material to Blender

## Usage
The system can be used by running the main workflow script with various options:
```
python material_generation_workflow.py [options]
```

Options include specifying model paths, output paths, sampling parameters, and whether to generate textures.