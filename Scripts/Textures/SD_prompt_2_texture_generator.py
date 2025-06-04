import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFilter
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load base SD pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to(device)

def generate_texture(prompt: str, output_dir: str = "./generated_textures", seed: int = 42):
    torch.manual_seed(seed)
    output_dir = os.path.join(output_dir, prompt)
    os.makedirs(output_dir, exist_ok=True)

    # Generate base color
    print("[+] Generating base color...")
    image = pipe(prompt, guidance_scale=7.5).images[0]
    base_path = os.path.join(output_dir, "base_color.png")
    image.save(base_path)

    # Generate grayscale variations for roughness/metallic
    roughness = image.convert("L")
    metallic = image.convert("L")
    normal = image.filter(ImageFilter.CONTOUR)  # Placeholder, real normal requires model

    roughness.save(os.path.join(output_dir, "roughness.png"))
    metallic.save(os.path.join(output_dir, "metallic.png"))
    normal.save(os.path.join(output_dir, "normal.png"))

    print(f"[✓] Textures saved to: {output_dir}")

# Example:
generate_texture("rusty metallic sci-fi panel")
generate_texture("seamless tiger stripes")
generate_texture("cute fox")