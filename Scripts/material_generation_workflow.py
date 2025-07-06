#!/usr/bin/env python3
"""
Material Generation Workflow

This script orchestrates the entire workflow for generating materials for Blender:
1. Predict nodes using node_autoregression.py
2. Predict edges using gnn_edge_sampler.py
3. Predict parameters using test_param_predictor.py
4. Generate textures using SD_CLIP_guided_texture_generator.py (optional)
5. Import the material to Blender using import_predicted_material.py

Usage:
    python material_generation_workflow.py [options]

Options:
    --id2node-json PATH       Path to id_to_node.json (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json)
    --model-in PATH           Path to node generator model (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Nodes/node_generator_mps.pth)
    --edge-model PATH         Path to edge predictor model (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/gnn_edge_model.pt)
    --param-model PATH        Path to parameter predictor model (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Parameters/param_predictor.pth)
    --output-json PATH        Path to save the predicted material graph (default: /Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Generated/predicted_material_graph.json)
    --material-name NAME      Name of the material to create in Blender (default: "Generated_Material")
    --num-samples INT         Number of node sequences to generate (default: 1)
    --max-len INT             Maximum length of generated node sequences (default: 64)
    --top-p FLOAT             Top-p sampling parameter (default: 0.9)
    --threshold FLOAT         Edge prediction threshold (default: 0.95)
    --blender-path PATH       Path to Blender executable (default: /Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender)
    --skip-blender            Skip importing to Blender (default: False)

    # Texture generation options
    --generate-textures       Generate textures for IMAGE TEX nodes (default: False)
    --texture-prompt TEXT     Prompt for texture generation (default: "PBR texture, uniform lighting")
    --texture-variants INT    Number of texture variants to generate (default: 3)
    --texture-output-dir PATH Directory to save generated textures (default: "./Scripts/Textures/generated_textures")
"""

import argparse
import json
import os
import subprocess
import sys
import torch
from pathlib import Path
import torch_geometric
from torch_geometric.data import Data

# Add parent directory to path so we can import from other scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from other scripts
from Scripts.Nodes.node_autoregression import NodeGenerator
from Scripts.Edges.gnn_edge_predictor import GNNModel, NUM_NODE_TYPES, NUM_SOCKET_TYPES
from Scripts.Parameters.gnn_edge_and_param_predictor import compute_param_stats, ParamDataset, MultiHeadParamPredictor
from Scripts.Textures.texture_generator import generate_textures

def parse_arguments():
    parser = argparse.ArgumentParser(description="Material Generation Workflow")

    # Node prediction arguments
    parser.add_argument("--id2node-json", default="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_node.json",
                        help="Path to id_to_node.json")
    parser.add_argument("--model-in", default="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Nodes/node_generator_mps.pth",
                        help="Path to node generator model")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of node sequences to generate")
    parser.add_argument("--max-len", type=int, default=64,
                        help="Maximum length of generated node sequences")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter")

    # Model dimensions (should match training)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)

    # Edge prediction arguments
    parser.add_argument("--edge-model", default="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Edges/gnn_edge_model.pt",
                        help="Path to edge predictor model")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Edge prediction threshold")
    parser.add_argument("--socket-map", default="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/id_to_socket.json",
                        help="Path to id_to_socket.json")

    # Parameter prediction arguments
    parser.add_argument("--param-model", default="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Parameters/param_predictor.pth",
                        help="Path to parameter predictor model")
    parser.add_argument("--param-json", default="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/merged_dataset.json",
                        help="Path to parameter dataset JSON")
    parser.add_argument("--node-type-to-params", default="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Auxiliary/node_type_to_params.json",
                        help="Path to node_type_to_params.json")

    # Output arguments
    parser.add_argument("--output-json", default="/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Generated/predicted_material_graph.json",
                        help="Path to save the predicted material graph")
    parser.add_argument("--material-name", default="Generated_Material",
                        help="Name of the material to create in Blender")

    # Texture generation arguments
    parser.add_argument("--generate-textures", action="store_true",
                        help="Generate textures for IMAGE TEX nodes")
    parser.add_argument("--texture-prompt", default="PBR texture, uniform lighting",
                        help="Prompt for texture generation")
    parser.add_argument("--texture-variants", type=int, default=1,
                        help="Number of texture variants to generate")
    parser.add_argument("--texture-output-dir", default="./Scripts/Textures/generated_textures",
                        help="Directory to save generated textures")

    # Blender arguments
    parser.add_argument("--blender-path", default="/Volumes/ProgramFiles/Apps/Blender_36.app/Contents/MacOS/Blender",
                        help="Path to Blender executable")
    parser.add_argument("--skip-blender", action="store_true",
                        help="Skip importing to Blender")

    return parser.parse_args()

def predict_nodes(args):
    """
    Predict node sequences using node_autoregression.py
    """
    print("\n" + "="*50)
    print("STEP 1: PREDICTING NODES")
    print("="*50)

    # device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Sampling on", device)

    # load id2node to know BOS/EOS, N
    id2node = json.load(Path(args.id2node_json).open("r"))
    numeric_ids = sorted(int(k) for k in id2node.keys())
    N = max(numeric_ids)
    vocab_size = N + 3
    BOS_ID = N + 1
    EOS_ID = N + 2

    # build model & load
    model = NodeGenerator(
        vocab_size = vocab_size,
        d_model    = args.d_model,
        nhead      = args.nhead,
        nlayers    = args.nlayers,
        max_seq_len= args.max_seq_len
    )
    state = torch.load(args.model_in, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    # generate
    best_sequence = None
    for i in range(args.num_samples):
        seq = torch.tensor([[BOS_ID]], device=device)
        with torch.no_grad():
            for _ in range(args.max_len):
                logits = model(seq)[-1,0]
                vals, idxs = torch.sort(logits, descending=True)
                probs = torch.nn.functional.softmax(vals, dim=0)
                cum   = probs.cumsum(0)
                k     = (~(cum>args.top_p)).sum().item() + 1
                choices     = idxs[:k]
                choice_probs= torch.nn.functional.softmax(vals[:k], dim=0)
                pick = choices[torch.multinomial(choice_probs,1)].item()
                seq = torch.cat([seq, torch.tensor([[pick]], device=device)], dim=0)
                if pick==EOS_ID: break

        tokens = seq.squeeze(1).tolist()
        names = [id2node[str(t)] for t in tokens if 1<=t<=N]
        print(f"\nSample {i+1}: {' → '.join(names)}", '\n', tokens)

        # Keep the first sequence (or we could implement some selection logic here)
        if i == 0:
            best_sequence = [t for t in tokens if 1<=t<=N]

    print("\nSelected node sequence:", best_sequence)

    # Ensure the sequence contains a Material Output node and at least one shader node
    material_output_id = 43  # OUTPUT_MATERIAL
    shader_node_ids = [6, 7, 8, 9, 10, 11, 12, 13, 24, 25, 28, 38, 52]  # Various shader nodes

    # Check if Material Output node exists
    has_material_output = material_output_id in best_sequence

    # Check if at least one shader node exists
    has_shader = any(shader_id in best_sequence for shader_id in shader_node_ids)

    # If not, add the missing nodes
    modified_sequence = best_sequence.copy()

    if not has_material_output:
        print("Adding Material Output node (OUTPUT_MATERIAL) to the sequence")
        modified_sequence.append(material_output_id)

    if not has_shader:
        # Add a Principled BSDF shader (most commonly used)
        principled_bsdf_id = 9  # BSDF_PRINCIPLED
        print("Adding Principled BSDF shader node (BSDF_PRINCIPLED) to the sequence")
        modified_sequence.append(principled_bsdf_id)

    if modified_sequence != best_sequence:
        print("\nModified node sequence to ensure Material Output and shader nodes:", modified_sequence)
        best_sequence = modified_sequence

    return best_sequence

def predict_edges(args, node_sequence):
    """
    Predict edges using gnn_edge_sampler.py
    """
    print("\n" + "="*50)
    print("STEP 2: PREDICTING EDGES")
    print("="*50)

    # Load mappings
    with open(args.id2node_json, "r") as f:
        id_to_node = json.load(f)
    with open(args.socket_map, "r") as f:
        id_to_socket = json.load(f)

    node_names = {int(k): v for k, v in id_to_node.items()}
    socket_names = {int(k): v for k, v in id_to_socket.items()}

    # Build input graph
    node_types = torch.tensor(node_sequence, dtype=torch.long)
    x = torch.nn.functional.one_hot(node_types, NUM_NODE_TYPES).float()

    edge_index = [
        [i, j] for i in range(len(node_sequence)) for j in range(len(node_sequence)) if i != j
    ]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    edge_type_pair = torch.tensor([
        [node_sequence[src], node_sequence[dst]] for src, dst in edge_index.t().tolist()
    ], dtype=torch.long)

    edge_distance = torch.tensor([
        [abs(src - dst)] for src, dst in edge_index.t().tolist()
    ], dtype=torch.float32)

    edge_attr = torch.zeros((edge_index.size(1), 2), dtype=torch.long)
    edge_exists = torch.zeros((edge_index.size(1),), dtype=torch.float32)
    socket_mask = torch.zeros_like(edge_exists)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_exists=edge_exists,
        socket_mask=socket_mask,
        edge_type_pair=edge_type_pair,
        edge_distance=edge_distance
    )

    # Load model and predict
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GNNModel(input_dim=NUM_NODE_TYPES, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(args.edge_model, map_location=device))
    model.eval()

    data = data.to(device)
    with torch.no_grad():
        edge_logits, socket_pred = model(data)
        edge_probs = torch.sigmoid(edge_logits)
        socket_pred = socket_pred.view(-1, 2, NUM_SOCKET_TYPES)
        src_sock = socket_pred[:, 0, :].argmax(dim=1).cpu().tolist()
        dst_sock = socket_pred[:, 1, :].argmax(dim=1).cpu().tolist()
        edge_probs = edge_probs.cpu().tolist()
        edge_list = data.edge_index.t().cpu().tolist()

    # Output predictions
    output_data = {
        "node_sequence": node_sequence,
        "edges": []
    }

    # First, add predicted edges with high probability
    for idx, (src, dst) in enumerate(edge_list):
        if edge_probs[idx] > args.threshold:
            output_data["edges"].append({
                "src_idx": src,
                "dst_idx": dst,
                "src_socket": src_sock[idx],
                "dst_socket": dst_sock[idx],
                "prob": edge_probs[idx]
            })

    # Track which nodes are connected
    connected_nodes = set()
    for edge in output_data["edges"]:
        connected_nodes.add(edge["src_idx"])
        connected_nodes.add(edge["dst_idx"])

    # Find nodes that are not connected
    unconnected_nodes = set(range(len(node_sequence))) - connected_nodes

    # Find shader nodes and material output node
    material_output_idx = None
    shader_node_indices = []
    tex_image_indices = []
    tex_coord_indices = []
    mapping_indices = []

    for idx, node_id in enumerate(node_sequence):
        node_type = node_names.get(node_id, "Unknown")
        if node_type == "OUTPUT_MATERIAL":
            material_output_idx = idx
        elif node_type in ["BSDF_PRINCIPLED", "BSDF_DIFFUSE", "BSDF_GLASS", "BSDF_ANISOTROPIC", 
                          "BSDF_REFRACTION", "BSDF_TOON", "BSDF_TRANSLUCENT", "BSDF_TRANSPARENT",
                          "EEVEE_SPECULAR", "EMISSION", "MIX_SHADER", "ADD_SHADER", "SUBSURFACE_SCATTERING"]:
            shader_node_indices.append(idx)
        elif node_type == "TEX_IMAGE":
            tex_image_indices.append(idx)
        elif node_type == "TEX_COORD":
            tex_coord_indices.append(idx)
        elif node_type == "MAPPING":
            mapping_indices.append(idx)

    # Ensure shader is connected to material output
    if material_output_idx is not None and shader_node_indices:
        # Check if there's already a connection between a shader and material output
        shader_to_output_exists = any(
            edge["src_idx"] in shader_node_indices and edge["dst_idx"] == material_output_idx
            for edge in output_data["edges"]
        )

        if not shader_to_output_exists:
            # Connect the first shader to material output
            print(f"Adding connection from shader node {shader_node_indices[0]} to Material Output node {material_output_idx}")
            output_data["edges"].append({
                "src_idx": shader_node_indices[0],
                "dst_idx": material_output_idx,
                "src_socket": 0,  # Assuming 0 is the shader output socket
                "dst_socket": 0,  # Assuming 0 is the surface input socket on Material Output
                "prob": 1.0
            })
            # Update connected nodes
            connected_nodes.add(shader_node_indices[0])
            connected_nodes.add(material_output_idx)
            # Remove from unconnected if they were there
            unconnected_nodes.discard(shader_node_indices[0])
            unconnected_nodes.discard(material_output_idx)

    # Handle Image Texture nodes - ensure they're in the chain: Texture Coordinate -> Mapping -> Image Texture
    # First, check if we need to add missing nodes to the sequence
    mapping_node_id = 33  # MAPPING
    tex_coord_node_id = 56  # TEX_COORD

    # Add missing nodes to the sequence if needed
    modified_sequence = node_sequence.copy()
    node_indices_updated = False

    # If we have image textures but no mapping nodes, add one
    if tex_image_indices and not mapping_indices:
        print("Adding Mapping node to the sequence for Image Texture nodes")
        modified_sequence.append(mapping_node_id)
        # New index will be at the end of the sequence
        new_mapping_idx = len(modified_sequence) - 1
        mapping_indices.append(new_mapping_idx)
        node_indices_updated = True

    # If we have mapping nodes but no texture coordinate nodes, add one
    if mapping_indices and not tex_coord_indices:
        print("Adding Texture Coordinate node to the sequence for Mapping nodes")
        modified_sequence.append(tex_coord_node_id)
        # New index will be at the end of the sequence
        new_texcoord_idx = len(modified_sequence) - 1
        tex_coord_indices.append(new_texcoord_idx)
        node_indices_updated = True

    # If we modified the sequence, update the node_sequence in output_data
    if node_indices_updated:
        output_data["node_sequence"] = modified_sequence
        print("Updated node sequence with missing nodes:", modified_sequence)

    # Now connect the nodes in the chain
    # Track which mapping nodes are connected to image textures
    mapping_connected_to_image = set()

    for tex_image_idx in tex_image_indices:
        # Check if this texture node is already connected to a mapping node
        tex_has_mapping = False
        connected_mapping_idx = None

        for edge in output_data["edges"]:
            if edge["src_idx"] in mapping_indices and edge["dst_idx"] == tex_image_idx:
                tex_has_mapping = True
                connected_mapping_idx = edge["src_idx"]
                mapping_connected_to_image.add(connected_mapping_idx)
                break

        # If not connected to mapping, connect it to an available mapping node
        if not tex_has_mapping and mapping_indices:
            # Try to use a mapping node that isn't already connected to an image texture
            available_mapping = [idx for idx in mapping_indices if idx not in mapping_connected_to_image]

            # If all mapping nodes are already connected, just use the first one
            mapping_idx = available_mapping[0] if available_mapping else mapping_indices[0]

            print(f"Adding connection from Mapping node {mapping_idx} to Image Texture node {tex_image_idx}")
            output_data["edges"].append({
                "src_idx": mapping_idx,
                "dst_idx": tex_image_idx,
                "src_socket": 0,  # Vector output
                "dst_socket": 0,  # Vector input
                "prob": 1.0
            })
            # Update connected nodes
            connected_nodes.add(mapping_idx)
            connected_nodes.add(tex_image_idx)
            # Remove from unconnected if they were there
            unconnected_nodes.discard(mapping_idx)
            unconnected_nodes.discard(tex_image_idx)
            # Mark this mapping node as connected to an image texture
            mapping_connected_to_image.add(mapping_idx)

    # Connect mapping nodes to texture coordinate nodes
    for mapping_idx in mapping_indices:
        # Check if this mapping node is already connected to a texture coordinate node
        mapping_has_texcoord = any(
            edge["src_idx"] in tex_coord_indices and edge["dst_idx"] == mapping_idx
            for edge in output_data["edges"]
        )

        # If not connected to texture coordinate, connect it
        if not mapping_has_texcoord and tex_coord_indices:
            texcoord_idx = tex_coord_indices[0]  # Use the first texture coordinate node
            print(f"Adding connection from Texture Coordinate node {texcoord_idx} to Mapping node {mapping_idx}")
            output_data["edges"].append({
                "src_idx": texcoord_idx,
                "dst_idx": mapping_idx,
                "src_socket": 0,  # UV output
                "dst_socket": 0,  # Vector input
                "prob": 1.0
            })
            # Update connected nodes
            connected_nodes.add(texcoord_idx)
            connected_nodes.add(mapping_idx)
            # Remove from unconnected if they were there
            unconnected_nodes.discard(texcoord_idx)
            unconnected_nodes.discard(mapping_idx)

    # Ensure all mapping nodes are connected to at least one image texture node
    for mapping_idx in mapping_indices:
        # Check if this mapping node is connected to any image texture node
        mapping_connected_to_image = any(
            edge["src_idx"] == mapping_idx and edge["dst_idx"] in tex_image_indices
            for edge in output_data["edges"]
        )

        # If not connected to any image texture, try to connect it to one
        if not mapping_connected_to_image and tex_image_indices:
            # Find an image texture that doesn't already have a mapping connection
            available_tex_images = []
            for tex_idx in tex_image_indices:
                has_mapping = any(
                    edge["src_idx"] in mapping_indices and edge["dst_idx"] == tex_idx
                    for edge in output_data["edges"]
                )
                if not has_mapping:
                    available_tex_images.append(tex_idx)

            # If all image textures already have mapping connections, just connect to the first one
            tex_image_idx = available_tex_images[0] if available_tex_images else tex_image_indices[0]

            print(f"Adding connection from Mapping node {mapping_idx} to Image Texture node {tex_image_idx}")
            output_data["edges"].append({
                "src_idx": mapping_idx,
                "dst_idx": tex_image_idx,
                "src_socket": 0,  # Vector output
                "dst_socket": 0,  # Vector input
                "prob": 1.0
            })
            # Update connected nodes
            connected_nodes.add(mapping_idx)
            connected_nodes.add(tex_image_idx)
            # Remove from unconnected if they were there
            unconnected_nodes.discard(mapping_idx)
            unconnected_nodes.discard(tex_image_idx)

    # Connect any remaining unconnected nodes to something
    if unconnected_nodes:
        print(f"Found {len(unconnected_nodes)} unconnected nodes, connecting them to the graph")

        # For each unconnected node, find a suitable node to connect to
        for node_idx in unconnected_nodes:
            node_id = node_sequence[node_idx]
            node_type = node_names.get(node_id, "Unknown")

            # Try to find a meaningful connection based on node type
            if node_type in ["BSDF_PRINCIPLED", "BSDF_DIFFUSE", "BSDF_GLASS", "BSDF_ANISOTROPIC", 
                            "BSDF_REFRACTION", "BSDF_TOON", "BSDF_TRANSLUCENT", "BSDF_TRANSPARENT",
                            "EEVEE_SPECULAR", "EMISSION", "MIX_SHADER", "ADD_SHADER", "SUBSURFACE_SCATTERING"]:
                # Shader node - connect to material output if available
                if material_output_idx is not None:
                    print(f"Connecting shader node {node_idx} to Material Output node {material_output_idx}")
                    output_data["edges"].append({
                        "src_idx": node_idx,
                        "dst_idx": material_output_idx,
                        "src_socket": 0,  # Shader output
                        "dst_socket": 0,  # Surface input
                        "prob": 1.0
                    })
                    continue

            # Special handling for Image Texture nodes - connect to shader inputs if possible
            elif node_type == "TEX_IMAGE":
                # Try to connect to a shader node input (like Base Color on Principled BSDF)
                if shader_node_indices:
                    shader_idx = shader_node_indices[0]  # Use the first shader node
                    # Common socket for color/texture input is 0 (Base Color for Principled BSDF)
                    print(f"Connecting Image Texture node {node_idx} to Shader node {shader_idx}")
                    output_data["edges"].append({
                        "src_idx": node_idx,
                        "dst_idx": shader_idx,
                        "src_socket": 0,  # Color output
                        "dst_socket": 0,  # Base Color input (for Principled BSDF)
                        "prob": 1.0
                    })
                    continue

            # For other node types or if no specific connection was made, connect to any other node
            # Prefer connecting to nodes that are already connected
            potential_targets = list(connected_nodes - {node_idx})
            if potential_targets:
                target_idx = potential_targets[0]  # Just pick the first one
                print(f"Connecting node {node_idx} to node {target_idx}")
                output_data["edges"].append({
                    "src_idx": node_idx,
                    "dst_idx": target_idx,
                    "src_socket": 0,  # Default output socket
                    "dst_socket": 0,  # Default input socket
                    "prob": 1.0
                })

    # Save to file
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Saved predicted material graph to {args.output_json}")
    print(f"Nodes: {len(node_sequence)}, Edges: {len(output_data['edges'])}")

    return output_data

def predict_parameters(args, graph_data):
    """
    Predict parameters using test_param_predictor.py
    """
    print("\n" + "="*50)
    print("STEP 3: PREDICTING PARAMETERS")
    print("="*50)

    # Load mappings and data
    param_ranges, dropdown_classes, checkbox_classes = compute_param_stats(args.param_json)

    with open(args.node_type_to_params) as f:
        node_type_to_params = json.load(f)

    with open(args.id2node_json) as f:
        id_to_node = {int(k): v for k, v in json.load(f).items()}

    node_sequence = graph_data["node_sequence"]

    # Create dummy dataset for parameter metadata
    dummy_dataset = ParamDataset(args.param_json, param_ranges, dropdown_classes, checkbox_classes, node_type_to_params)
    param_types = dummy_dataset.param_types

    # Helper: Encode/Decode Param Keys
    def encode_param_key(key: str) -> str:
        return key.replace(".", "__")

    # Load model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultiHeadParamPredictor(
        input_dim=len(dummy_dataset[0][0]),
        param_keys=[encode_param_key(k) for k in dummy_dataset.param_keys],
        param_types={encode_param_key(k): v for k, v in param_types.items()},
        dropdown_classes={encode_param_key(k): v for k, v in dropdown_classes.items()},
        checkbox_classes={encode_param_key(k): v for k, v in checkbox_classes.items()}
    )
    model.load_state_dict(torch.load(args.param_model, map_location=device))
    model.to(device).eval()

    # Predict parameters per node
    print("\n🔍 Predicting parameter values for each node:")

    # Add parameters to graph data
    graph_data["parameters"] = {}

    EXCLUDED_PARAMS = {"Image Name", "Image Path"}

    for idx, node_id in enumerate(node_sequence):
        node_type = id_to_node.get(node_id, "Unknown")
        one_hot = [1.0 if t == node_type else 0.0 for t in dummy_dataset.node_type_list]
        x = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0).to(device)

        shared = model.shared(x)
        print(f"\nNode {idx} [{node_type}]:")

        # Store parameters for this node
        graph_data["parameters"][idx] = {"node_type": node_type, "params": {}}

        for param in node_type_to_params.get(node_type, []):
            if param in EXCLUDED_PARAMS:
                continue

            encoded = encode_param_key(param)
            head_type = model.param_types.get(encoded)
            if not head_type:
                continue

            if head_type == "reg" and encoded in model.regression_heads:
                pred = model.regression_heads[encoded](shared).squeeze().item()
                denorm = pred * (param_ranges[param][1] - param_ranges[param][0] + 1e-8) + param_ranges[param][0]
                print(f"  {param:<20} = {denorm:.4f}")
                graph_data["parameters"][idx]["params"][param] = denorm

            elif head_type == "cls" and encoded in model.classification_heads:
                logits = model.classification_heads[encoded](shared).squeeze()
                class_idx = torch.argmax(logits).item()
                label = dropdown_classes[encoded][class_idx]
                print(f"  {param:<20} = '{label}'")
                graph_data["parameters"][idx]["params"][param] = label

            elif head_type == "bin" and encoded in model.binary_heads:
                logits = model.binary_heads[encoded](shared).squeeze()
                class_idx = torch.argmax(logits).item()
                label = model.checkbox_classes[encoded][class_idx]
                print(f"  {param:<20} = {label}")
                graph_data["parameters"][idx]["params"][param] = label

    # Save updated graph data with parameters
    with open(args.output_json, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"\n✅ Updated predicted material graph with parameters at {args.output_json}")

    return graph_data

def generate_textures_for_nodes(args, graph_data):
    """
    Generate textures for IMAGE TEX nodes using texture_generator.py
    """
    if not args.generate_textures:
        return graph_data

    print("\n" + "="*50)
    print("STEP 4: GENERATING TEXTURES")
    print("="*50)

    # Load id_to_node mapping
    with open(args.id2node_json) as f:
        id_to_node = {int(k): v for k, v in json.load(f).items()}

    # Find all TEX_IMAGE nodes in the graph
    tex_image_nodes = []
    for idx, node_id in enumerate(graph_data["node_sequence"]):
        if id_to_node.get(node_id) == "TEX_IMAGE":
            tex_image_nodes.append(idx)

    if not tex_image_nodes:
        print("No IMAGE TEX nodes found in the material graph.")
        return graph_data

    print(f"Found {len(tex_image_nodes)} IMAGE TEX nodes in the material graph.")

    # Generate textures
    print(f"\nGenerating textures with prompt: '{args.texture_prompt}'")
    texture_maps = generate_textures(
        prompt=args.texture_prompt,
        output_dir=args.texture_output_dir,
        num_variants=args.texture_variants,
        seed=42
    )

    # Add texture paths to the graph data
    if "textures" not in graph_data:
        graph_data["textures"] = {}

    # Store texture paths in the graph data
    graph_data["textures"]["paths"] = texture_maps

    # Assign textures to TEX_IMAGE nodes
    for idx in tex_image_nodes:
        # Make sure parameters exist for this node
        if "parameters" not in graph_data:
            graph_data["parameters"] = {}

        if str(idx) not in graph_data["parameters"]:
            graph_data["parameters"][str(idx)] = {"node_type": "TEX_IMAGE", "params": {}}

        # Assign base color texture to the node
        graph_data["parameters"][str(idx)]["params"]["Image"] = texture_maps["base_color"]
        print(f"Assigned texture to node {idx}: {texture_maps['base_color']}")

    # Save updated graph data
    with open(args.output_json, "w") as f:
        json.dump(graph_data, f, indent=2)

    print(f"\n✅ Updated material graph with texture information at {args.output_json}")

    return graph_data

def import_to_blender(args):
    """
    Import the material to Blender using import_predicted_material.py
    """
    print("\n" + "="*50)
    print("STEP 5: IMPORTING TO BLENDER")
    print("="*50)

    blender_script = "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Scripts/Blender/import_predicted_material.py"

    # Build command
    cmd = [
        args.blender_path,
        "--python", blender_script,
        "--", args.material_name, args.output_json
    ]

    print(f"Running Blender with command: {' '.join(cmd)}")

    # Run Blender
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ Successfully imported material to Blender")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error importing material to Blender: {e}")
        return False

    return True

def main():
    args = parse_arguments()

    # Step 1: Predict nodes
    node_sequence = predict_nodes(args)

    # Step 2: Predict edges
    graph_data = predict_edges(args, node_sequence)

    # Step 3: Predict parameters
    graph_data = predict_parameters(args, graph_data)

    # Step 4: Generate textures (optional)
    if args.generate_textures:
        graph_data = generate_textures_for_nodes(args, graph_data)
    else:
        print("\n⚠️ Skipping texture generation as it's not enabled")

    # Step 5: Import to Blender (optional)
    if not args.skip_blender:
        import_to_blender(args)
    else:
        print("\n⚠️ Skipping Blender import as requested")

    print("\n" + "="*50)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Generated material graph saved to: {args.output_json}")
    if args.generate_textures:
        print(f"Textures generated and assigned to IMAGE TEX nodes")
    if not args.skip_blender:
        print(f"Material '{args.material_name}' created in Blender")

    return 0

if __name__ == "__main__":
    sys.exit(main())
