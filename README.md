# Blender Material Generator

This project uses machine learning to generate materials for Blender. It consists of several components:

1. **Node Prediction**: Predicts a sequence of nodes for a material using a transformer-based model.
2. **Edge Prediction**: Predicts edges between nodes using a Graph Neural Network (GNN).
3. **Parameter Prediction**: Predicts parameters for each node using a multi-head neural network.
4. **Texture Generation**: Generates textures for IMAGE TEX nodes using Stable Diffusion guided by CLIP.
5. **Blender Import**: Imports the generated material into Blender.

## Workflow

The workflow for generating materials is as follows:

1. Predict a sequence of nodes using the node autoregression model.
2. Predict edges between nodes using the GNN edge predictor.
3. Predict parameters for each node using the parameter predictor.
4. (Optional) Generate textures for IMAGE TEX nodes using Stable Diffusion guided by CLIP.
5. Import the material into Blender.

## Usage

### Automated Workflow

The entire workflow can be run automatically using the `material_generation_workflow.py` script:

```bash
python Scripts/material_generation_workflow.py [options]
```

#### Options

- `--id2node-json PATH`: Path to id_to_node.json (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json)
- `--model-in PATH`: Path to node generator model (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Nodes/node_generator_mps.pth)
- `--edge-model PATH`: Path to edge predictor model (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/gnn_edge_model.pt)
- `--param-model PATH`: Path to parameter predictor model (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Parameters/param_predictor.pth)
- `--output-json PATH`: Path to save the predicted material graph (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Generated/predicted_material_graph.json)
- `--material-name NAME`: Name of the material to create in Blender (default: "Generated_Material")
- `--num-samples INT`: Number of node sequences to generate (default: 1)
- `--max-len INT`: Maximum length of generated node sequences (default: 64)
- `--top-p FLOAT`: Top-p sampling parameter (default: 0.9)
- `--threshold FLOAT`: Edge prediction threshold (default: 0.95)
- `--blender-path PATH`: Path to Blender executable (default: /Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender)
- `--skip-blender`: Skip importing to Blender (default: False)

#### Texture Generation Options
- `--generate-textures`: Generate textures for IMAGE TEX nodes (default: False)
- `--texture-prompt TEXT`: Prompt for texture generation (default: "PBR texture, uniform lighting")
- `--texture-variants INT`: Number of texture variants to generate (default: 3)
- `--texture-output-dir PATH`: Directory to save generated textures (default: "./Scripts/Textures/generated_textures")

### Manual Workflow

Alternatively, each step can be run manually:

1. **Node Prediction**:
   ```bash
   python Scripts/Nodes/node_autoregression.py sample \
     --id2node-json /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json \
     --model-in /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Nodes/node_generator_mps.pth \
     --num-samples 5
   ```

2. **Edge Prediction**:
   ```bash
   python Scripts/Edges/gnn_edge_sampler.py
   ```

3. **Parameter Prediction**:
   ```bash
   python Scripts/Parameters/test_param_predictor.py
   ```

4. **Texture Generation** (optional):
   ```bash
   python Scripts/Textures/texture_generator.py --prompt "PBR texture of red brick wall" --output-dir ./Scripts/Textures/generated_textures
   ```

5. **Blender Import**:
   ```bash
   /Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender --python Scripts/Blender/import_predicted_material.py
   ```

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- Diffusers (for texture generation)
- Transformers (for texture generation)
- Pillow (for texture generation)
- Blender 3.6+

## Project Structure

- `Dataset/`: Contains datasets for training and testing.
  - `Auxiliary/`: Contains auxiliary files like mappings.
  - `Generated/`: Contains generated material graphs.
  - `Refined/`: Contains refined datasets.
- `Models/`: Contains trained models.
  - `Edges/`: Contains edge prediction models.
  - `Nodes/`: Contains node prediction models.
  - `Parameters/`: Contains parameter prediction models.
- `Scripts/`: Contains scripts for each step of the workflow.
  - `Blender/`: Contains scripts for importing materials to Blender.
  - `Edges/`: Contains scripts for edge prediction.
  - `Nodes/`: Contains scripts for node prediction.
  - `Parameters/`: Contains scripts for parameter prediction.
  - `Textures/`: Contains scripts for texture generation.
    - `generated_textures/`: Contains generated textures.
  - `material_generation_workflow.py`: Script for running the entire workflow.
