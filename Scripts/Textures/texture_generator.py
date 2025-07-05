import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageFilter
import os

def generate_textures(prompt, output_dir=None, num_variants=5, seed=42):
    """
    Generate textures using Stable Diffusion guided by CLIP.
    
    Args:
        prompt (str): The prompt to guide texture generation
        output_dir (str, optional): Directory to save generated textures. If None, a directory is created based on the prompt.
        num_variants (int, optional): Number of texture variants to generate. Defaults to 5.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        dict: Dictionary containing paths to generated texture maps
    """
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Set output directory
    if output_dir is None:
        output_dir = f"./generated_textures/{prompt.replace(' ', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("[+] Loading models...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "dream-textures/texture-diffusion",
        safety_checker=None,
        feature_extractor=None,
    )
    pipe = pipe.to(device)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)
    
    # Generate candidate images
    print("[+] Generating texture variants...")
    image_paths = []
    for i in range(num_variants):
        torch.manual_seed(seed + i)
        image = pipe(prompt, guidance_scale=7.5).images[0]
        path = os.path.join(output_dir, f"variant_{i+1}.png")
        image.save(path)
        image_paths.append(path)
    
    # CLIP ranking
    print("[+] Ranking with CLIP...")
    scores = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = clip_model(**inputs)
            similarity = outputs.logits_per_image.softmax(dim=1)[0][0].item()
            scores.append((path, similarity))
    
    # Sort by similarity score
    best_path, best_score = sorted(scores, key=lambda x: x[1], reverse=True)[0]
    best_image = Image.open(best_path)
    print(f"Best match: {os.path.basename(best_path)} (score={best_score:.4f})")
    
    # Save as base_color + other maps
    texture_maps = {}
    
    base_name = os.path.join(output_dir, "base_color.png")
    best_image.save(base_name)
    texture_maps["base_color"] = base_name
    
    roughness = best_image.convert("L")
    roughness_path = os.path.join(output_dir, "roughness.png")
    roughness.save(roughness_path)
    texture_maps["roughness"] = roughness_path
    
    metallic = best_image.convert("L")
    metallic_path = os.path.join(output_dir, "metallic.png")
    metallic.save(metallic_path)
    texture_maps["metallic"] = metallic_path
    
    normal = best_image.filter(ImageFilter.CONTOUR)
    normal_path = os.path.join(output_dir, "normal.png")
    normal.save(normal_path)
    texture_maps["normal"] = normal_path
    
    print(f"[✓] Saved all maps in: {output_dir}")
    return texture_maps

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate textures using Stable Diffusion guided by CLIP")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to guide texture generation")
    parser.add_argument("--output-dir", type=str, help="Directory to save generated textures")
    parser.add_argument("--num-variants", type=int, default=5, help="Number of texture variants to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generate_textures(
        prompt=args.prompt,
        output_dir=args.output_dir,
        num_variants=args.num_variants,
        seed=args.seed
    )