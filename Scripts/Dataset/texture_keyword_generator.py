#!/usr/bin/env python3
"""
────────────────────────────────────────────────────────────────────────────
TEXTURE LABELING USING CLIP (BLENDER MATERIAL GRAPH SUPPORT)
────────────────────────────────────────────────────────────────────────────

This script uses the OpenAI CLIP model (ViT-B/32) to analyze texture images
used in Blender material graphs and assign descriptive texture labels.

It compares each material's **Base Color** texture to reference prototypes from:
  • DTD (Describable Textures)
  • MINC-2500
  • FMD (Flickr Material Database)
  • Your own clusters (MyTexturesDataset: pattern_clusters & color_clusters)
  • Curated text prompts (e.g., "metallic", "stone", "plastic")

New switches:
  A) --prototype-mode {single,mean}
     - 'mean' averages multiple images per label to form a stronger prototype.
  B) --sim-mode {raw,spherical,zscore,zscore_diverse}
     - 'spherical' L2-normalizes (cosine). Recommended.
  C) 'zscore' and 'zscore_diverse' apply per-source z-score calibration;
     the latter also enforces a per-source cap inside top-k.

Base Color selection:
  1) filename heuristics (positive/negative tokens)
  2) content heuristics (colorfulness + saturation + blue-bias to avoid normals)

────────────────────────────────────────────────────────────────────────────
INPUTS:
- A material dataset (JSON) with Blender-style nodes and texture filenames.
- DTD dataset directory (images/<class>/*.jpg).
- MINC-2500 images directory (images/<class>/**/<image>.jpg).
- FMD dataset directory (image/<class>/*.jpg).
- Your clustered textures:
    MyTexturesDataset/pattern_clusters/<cluster_XX>/*.*
    MyTexturesDataset/color_clusters/<cluster_XX>/*.*
- A folder with all textures used by materials (albedo + PBR maps).
- CLIP model and processor (auto-downloaded from Hugging Face).

OUTPUTS:
- Updated JSON dataset with fields:
    • CLIP_source_file — chosen base color filename
    • CLIP_label       — best match
    • CLIP_top3        — top-k matching labels with similarity scores
- Optional visualization of the material vs. best match.

────────────────────────────────────────────────────────────────────────────
REQUIREMENTS:
  pip install torch torchvision transformers pillow numpy scikit-learn tqdm matplotlib

python texture_keyword_generator.py \
  --dataset-path "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Dataset/Refined/merged_dataset.json" \
  --textures-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Projects/textures" \
  --dtd-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/dtd/images" \
  --minc-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/minc/minc-2500/images" \
  --fmd-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/fmd/image" \
  --mtd-pattern-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/MyTexturesDataset/pattern_clusters" \
  --mtd-color-dir "/Volumes/Data/University/PPMGR/Blender_Mat_Generator_PPMGR/Models/Textures/MyTexturesDataset/color_clusters" \
  --prototype-mode mean \
  --sim-mode spherical \
  --limit-materials 15 --top-k 10
────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import math
import argparse
from itertools import islice
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ================================
# I/O
# ================================
def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_dataset(dataset, path):
    with open(path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Saved updated dataset to {path}")

# ================================
# Base Color selection
# ================================
POSITIVE_KEYS = [
    "basecolor", "base_color", "albedo", "diffuse", "albd",
    "_col", "-col", " color", "colour", "base-col", "base col", "beauty"
]
NEGATIVE_KEYS = [
    "normal", "_nrm", "-nrm", "nrm", "nor", "_n", "-n",
    "height", "disp", "displ", "displace", "bump", "dnm",
    "rough", "roughness", "gloss", "spec", "specular", "metal", "metallic",
    "ao", "ambientocclusion", "cavity", "mask", "opacity", "alpha", "emiss", "emission",
    "deriv", "ddx", "ddy", "dx", "dy"
]

def _has_any(s, keys): s=s.lower(); return any(k in s for k in keys)
def _positive_score(name): n=name.lower(); return sum(1 for k in POSITIVE_KEYS if k in n)
def _negative_hit(name): return _has_any(name, NEGATIVE_KEYS)

def _pil_to_np(img): return np.asarray(img).astype(np.float32)

def _colorfulness_hs(img):
    arr = _pil_to_np(img)
    if arr.ndim == 2: return 0.0
    R,G,B = arr[...,0], arr[...,1], arr[...,2]
    rg = R - G
    yb = 0.5*(R+G) - B
    return math.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3*math.sqrt(np.mean(np.abs(rg))**2 + np.mean(np.abs(yb))**2)

def _mean_saturation(img):
    hsv = img.convert("HSV")
    s = np.asarray(hsv)[...,1].astype(np.float32)/255.0
    return float(np.mean(s))

def _is_probable_normal(img):
    arr = _pil_to_np(img)
    if arr.ndim == 2: return False
    mean_rgb = arr.reshape(-1,3).mean(axis=0)
    b,g,r = mean_rgb[2], mean_rgb[1], mean_rgb[0]
    sat = _mean_saturation(img)
    bluish = (b > r + 15) and (b > g + 15)
    return bluish and (0.1 <= sat <= 0.6)

def _is_probable_grayscale(img): return _mean_saturation(img) < 0.05

def _score_basecolor_candidate(path):
    name = os.path.basename(path)
    pos = _positive_score(name)
    try: img = Image.open(path).convert("RGB")
    except Exception: return (pos, -1.0, -1.0, -1.0, -1.0)
    cf = _colorfulness_hs(img); sat = _mean_saturation(img)
    normal_pen = 1.0 if _is_probable_normal(img) else 0.0
    gray_pen   = 1.0 if _is_probable_grayscale(img) else 0.0
    return (pos, cf, sat, -normal_pen, -gray_pen)

def choose_basecolor_image(paths):
    if not paths: return None
    filtered = [p for p in paths if not _negative_hit(os.path.basename(p))]
    candidates = filtered if filtered else paths
    scored = [( _score_basecolor_candidate(p), p ) for p in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

# ================================
# Reference library (A-mode capable)
# ================================
def _collect_images_in_class(label_path, recursive, max_images_per_class):
    candidates = []
    if recursive:
        for root, _, files in os.walk(label_path):
            imgs = [os.path.join(root,f) for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
            candidates.extend(imgs)
    else:
        candidates = [os.path.join(label_path,f) for f in os.listdir(label_path)
                      if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if max_images_per_class > 0:
        candidates = candidates[:max_images_per_class]
    return candidates

def _embed_image(img_path, processor, model, device):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs).cpu().numpy()[0]
    return emb

def _mean_embed(paths, processor, model, device):
    embs = []
    for p in paths:
        try: embs.append(_embed_image(p, processor, model, device))
        except Exception as e: print(f"[Warn] skip {p}: {e}")
    if not embs:
        return None
    return np.mean(np.stack(embs, axis=0), axis=0)

def add_reference_images(folder, prefix, recursive, processor, model, device,
                         prototype_mode="mean", max_images_per_class=5):
    """
    prefix: "DTD" | "MINC" | "FMD" | "MyTexturesDataset_pattern" | "MyTexturesDataset_color"
    prototype_mode: 'mean' (A) or 'single'
    """
    emb_dict = defaultdict(list)
    if not os.path.isdir(folder):
        print(f"[{prefix}] WARNING: folder does not exist: {folder}")
        return emb_dict, 0

    count = 0
    for label in sorted(os.listdir(folder)):
        label_path = os.path.join(folder, label)
        if not os.path.isdir(label_path):
            continue
        candidates = _collect_images_in_class(label_path, recursive, max_images_per_class)
        if not candidates: continue

        if prototype_mode == "mean":
            mean_emb = _mean_embed(candidates, processor, model, device)
            if mean_emb is not None:
                emb_dict[f"{prefix}:{label}"].append(mean_emb)
                count += len(candidates)
        else:
            # just take the first readable image
            try:
                emb = _embed_image(candidates[0], processor, model, device)
                emb_dict[f"{prefix}:{label}"].append(emb); count += 1
            except Exception as e:
                print(f"[Warn] {prefix}:{label} single embed failed: {e}")

    print(f"[{prefix}] Loaded {count} embeddings ({prototype_mode}).")
    return emb_dict, count

def load_reference_library(dtd_dir, minc_dir, fmd_dir, mtd_p_dir, mtd_c_dir,
                           processor, model, device,
                           prototype_mode="mean",  # A
                           max_images_per_class=5):
    emb_dict = defaultdict(list)

    # DTD
    d, _ = add_reference_images(dtd_dir, "DTD", False, processor, model, device,
                                prototype_mode, max_images_per_class)
    emb_dict.update(d)
    # MINC (recursive)
    d, _ = add_reference_images(minc_dir, "MINC", True, processor, model, device,
                                prototype_mode, max_images_per_class)
    for k, v in d.items(): emb_dict[k].extend(v)
    # FMD
    d, _ = add_reference_images(fmd_dir, "FMD", False, processor, model, device,
                                prototype_mode, max_images_per_class)
    for k, v in d.items(): emb_dict[k].extend(v)
    # Your pattern clusters
    d, _ = add_reference_images(mtd_p_dir, "MyTexturesDataset_pattern", False, processor, model, device,
                                prototype_mode, max_images_per_class)
    for k, v in d.items(): emb_dict[k].extend(v)
    # Your color clusters
    d, _ = add_reference_images(mtd_c_dir, "MyTexturesDataset_color", False, processor, model, device,
                                prototype_mode, max_images_per_class)
    for k, v in d.items(): emb_dict[k].extend(v)

    # Curated text prompts
    curated_prompts = [
        "wooden", "metallic", "fabric", "plastic", "concrete", "rock", "stone",
        "leather", "painted", "dirty", "clean", "smooth", "rough", "brushed metal"
    ]
    for prompt in curated_prompts:
        inputs = processor(text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs).cpu().numpy()[0]
        emb_dict[f"Prompt:{prompt}"].append(emb)
    print(f"[Prompt] Loaded {len(curated_prompts)} text prompts.")

    # Average per label to a single prototype vector
    ref_labels, ref_embeddings = [], []
    for label, embs in emb_dict.items():
        mean_emb = np.mean(np.stack(embs, axis=0), axis=0)
        ref_labels.append(label)
        ref_embeddings.append(mean_emb)
    ref_embeddings = np.array(ref_embeddings)

    print(f"[Summary] Total label prototypes: {len(ref_labels)} "
          f"(DTD/MINC/FMD/MyTexturesDataset --- Prompt merged)")
    return ref_embeddings, ref_labels

# ================================
# Similarity strategies (B & C)
# ================================
SOURCE_WEIGHTS = {
    # Tweak if you want a small boost to your own clusters, etc.
    "DTD": 1.00, "MINC": 0.98, "FMD": 1.00,
    "MyTexturesDataset_pattern": 1.05, "MyTexturesDataset_color": 1.05,
    "Prompt": 0.98
}

def l2_normalize_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def apply_source_weights(sims, ref_labels):
    sims = np.asarray(sims).copy()
    for i, lab in enumerate(ref_labels):
        src = lab.split(":",1)[0] if ":" in lab else "Unknown"
        sims[i] *= SOURCE_WEIGHTS.get(src, 1.0)
    return sims

def rerank_with_source_zscore(sims, ref_labels):
    sims = np.asarray(sims)
    sims_adj = sims.copy()
    by_src = defaultdict(list)
    for i, lab in enumerate(ref_labels):
        src = lab.split(":",1)[0] if ":" in lab else "Unknown"
        by_src[src].append(i)
    for src, idxs in by_src.items():
        vals = sims[idxs]
        mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
        sims_adj[idxs] = (vals - mu) / sd
    return sims_adj

def topk_diverse(sims_adj, ref_labels, k=10, max_per_source=3):
    order = np.argsort(sims_adj)[::-1]
    chosen, per_src = [], defaultdict(int)
    for i in order:
        src = ref_labels[i].split(":",1)[0] if ":" in ref_labels[i] else "Unknown"
        if per_src[src] >= max_per_source: continue
        chosen.append(i); per_src[src]+=1
        if len(chosen) == k: break
    return chosen

# ================================
# Prediction
# ================================
def predict_clip_nn(img_path, model, processor, ref_embeds_raw, ref_labels, device,
                    top_k=10, sim_mode="spherical", max_per_source=3):
    """
    sim_mode:
      - 'raw'        : plain cosine_similarity
      - 'spherical'  : L2-normalize refs & query, dot-product
      - 'zscore'     : per-source z-score calibration (on weighted sims)
      - 'zscore_diverse': z-score + per-source cap in top-k
    """
    img = Image.open(img_path).convert("RGB")
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    ])(img)

    inputs = processor(images=aug, return_tensors="pt").to(device)
    with torch.no_grad():
        q = model.get_image_features(**inputs).cpu().numpy()  # (1, D)

    if sim_mode == "raw":
        # cosine_similarity does its own norms
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(q, ref_embeds_raw)[0]
        idxs = np.argsort(sims)[-top_k:][::-1]

    elif sim_mode == "spherical":
        R = l2_normalize_rows(ref_embeds_raw)        # B
        qn = l2_normalize_rows(q)
        sims = (qn @ R.T)[0]                         # cosine via dot
        sims = apply_source_weights(sims, ref_labels)
        idxs = np.argsort(sims)[-top_k:][::-1]

    elif sim_mode in ("zscore", "zscore_diverse"):
        R = l2_normalize_rows(ref_embeds_raw)        # B baseline
        qn = l2_normalize_rows(q)
        sims = (qn @ R.T)[0]
        sims = apply_source_weights(sims, ref_labels)
        sims_adj = rerank_with_source_zscore(sims, ref_labels)  # C
        if sim_mode == "zscore_diverse":
            idxs = topk_diverse(sims_adj, ref_labels, k=top_k, max_per_source=max_per_source)
        else:
            idxs = np.argsort(sims_adj)[-top_k:][::-1]
    else:
        raise ValueError(f"Unknown sim_mode: {sim_mode}")

    top = [(ref_labels[i], float(sims[i])) for i in idxs]
    return top, img

# ================================
# Pipeline
# ================================
def process(dataset_path, dtd_dir, minc_dir, fmd_dir, mtd_p_dir, mtd_c_dir,
            textures_dir,
            prototype_mode="mean",   # A
            sim_mode="spherical",    # B/C selection
            top_k=10,
            max_per_source=3,
            max_images_per_class=5,
            limit_materials=15,
            output_path=None):

    dataset = load_dataset(dataset_path)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    # Safe loader (slow/fast fallback not strictly necessary here)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    ref_embeds_raw, ref_labels = load_reference_library(
        dtd_dir, minc_dir, fmd_dir, mtd_p_dir, mtd_c_dir,
        processor, model, device,
        prototype_mode=prototype_mode,          # A
        max_images_per_class=max_images_per_class
    )

    for name, mat in tqdm(islice(dataset["materials"].items(), limit_materials), desc=f"Processing first {limit_materials} materials"):
        tex_nodes = [
            node for node in mat["nodes"]
            if node.get("type") == "TEX_IMAGE" and "Image Name" in node.get("parameters", {})
        ]
        if not tex_nodes:
            continue

        # Collect candidate paths from shared textures_dir
        candidate_paths = []
        for node in tex_nodes:
            img_name = node["parameters"]["Image Name"]
            if not isinstance(img_name, str): continue
            p = os.path.join(textures_dir, img_name)
            if os.path.exists(p) and p.lower().endswith(('.jpg','.jpeg','.png')):
                candidate_paths.append(p)
        if not candidate_paths:
            continue

        basecolor_path = choose_basecolor_image(candidate_paths)
        if basecolor_path is None:
            continue

        top_matches, mat_img = predict_clip_nn(
            basecolor_path, model, processor, ref_embeds_raw, ref_labels, device,
            top_k=top_k, sim_mode=sim_mode, max_per_source=max_per_source
        )

        # annotate node that actually used basecolor_path, else first
        chosen_node = None
        for node in tex_nodes:
            if os.path.join(textures_dir, node["parameters"]["Image Name"]) == basecolor_path:
                chosen_node = node; break
        if chosen_node is None: chosen_node = tex_nodes[0]

        chosen_node["parameters"]["CLIP_source_file"] = os.path.basename(basecolor_path)
        chosen_node["parameters"]["CLIP_label"] = top_matches[0][0]
        chosen_node["parameters"]["CLIP_top3"] = [f"{lab} ({sc:.2f})" for lab, sc in top_matches[:3]]

        print(f"[{name}] {os.path.basename(basecolor_path)} → Top-{min(3,top_k)}: {chosen_node['parameters']['CLIP_top3']}")

        # Optional quick vis for best match
        try:
            best_label = top_matches[0][0]
            if ":" in best_label:
                source, label = best_label.split(":", 1)
                label_img_path = None
                # map back to an example image for display
                if source == "DTD":
                    class_dir = os.path.join(dtd_dir, label)
                    if os.path.isdir(class_dir):
                        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                        if files: label_img_path = os.path.join(class_dir, files[0])
                elif source == "MINC":
                    for root, _, files in os.walk(os.path.join(minc_dir, label)):
                        ims = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
                        if ims: label_img_path = os.path.join(root, ims[0]); break
                elif source == "FMD":
                    class_dir = os.path.join(fmd_dir, label)
                    if os.path.isdir(class_dir):
                        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                        if files: label_img_path = os.path.join(class_dir, files[0])
                elif source.startswith("MyTexturesDataset_"):
                    base_dir = mtd_p_dir if "pattern" in source else mtd_c_dir
                    class_dir = os.path.join(base_dir, label)
                    if os.path.isdir(class_dir):
                        files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
                        if files: label_img_path = os.path.join(class_dir, files[0])

                if label_img_path:
                    label_img = Image.open(label_img_path).convert("RGB")
                    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
                    axs[0].imshow(mat_img); axs[0].set_title(f"Material\n{os.path.basename(basecolor_path)}", fontsize=9)
                    axs[1].imshow(label_img); axs[1].set_title(f"Best Match: {label}", fontsize=9)
                    for ax in axs: ax.axis('off')
                    plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"[!] Visualization error for {basecolor_path}: {e}")

    save_dataset(dataset, output_path or dataset_path)
    return dataset

# ================================
# CLI
# ================================
def main():
    ap = argparse.ArgumentParser(description="CLIP-based texture keywording with A/B/C strategies.")
    ap.add_argument("--dataset-path", required=True, type=str)
    ap.add_argument("--textures-dir", required=True, type=str)

    ap.add_argument("--dtd-dir", required=True, type=str)
    ap.add_argument("--minc-dir", required=True, type=str)
    ap.add_argument("--fmd-dir", required=True, type=str)
    ap.add_argument("--mtd-pattern-dir", required=True, type=str)
    ap.add_argument("--mtd-color-dir", required=True, type=str)

    ap.add_argument("--prototype-mode", choices=["single","mean"], default="mean", help="A: build label prototypes from a single image or mean over multiple")
    ap.add_argument("--sim-mode", choices=["raw","spherical","zscore","zscore_diverse"], default="spherical",
                    help="B/C: similarity strategy")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-per-source", type=int, default=3, help="Used only for zscore_diverse")
    ap.add_argument("--max-images-per-class", type=int, default=5, help="Max images per class/cluster when building prototypes (mean mode)")
    ap.add_argument("--limit-materials", type=int, default=15, help="Just for quick runs; set big for full pass")
    ap.add_argument("--output-path", type=str, default=None)

    args = ap.parse_args()

    process(
        dataset_path=args.dataset_path,
        dtd_dir=args.dtd_dir,
        minc_dir=args.minc_dir,
        fmd_dir=args.fmd_dir,
        mtd_p_dir=args.mtd_pattern_dir,
        mtd_c_dir=args.mtd_color_dir,
        textures_dir=args.textures_dir,
        prototype_mode=args.prototype_mode,
        sim_mode=args.sim_mode,
        top_k=args.top_k,
        max_per_source=args.max_per_source,
        max_images_per_class=args.max_images_per_class,
        limit_materials=args.limit_materials,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()
