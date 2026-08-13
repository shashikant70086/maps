import os
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import CLIPVisionModelWithProjection

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
)


# ============================================================
# PATHS
# ============================================================

SATELLITE_PATH = "data/input/satellite.jpg"
STYLE_PATH = "data/output/style_reference.png"

OUTPUT_PATH = "data/output/phase1_controlnet.png"
CANNY_OUTPUT = "data/output/phase1_canny.png"


# ============================================================
# MODELS
# ============================================================

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0"

IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

# IMPORTANT:
# Lower = preserve the satellite more strongly.
STRENGTH = 0.20

# How strongly ControlNet follows the Canny structure.
CONTROLNET_SCALE = 0.65

# Style reference influence.
IP_ADAPTER_SCALE = 0.90

STEPS = 30
GUIDANCE = 5.0

SEED = 42


# ============================================================
# GENERATION SIZE
# ============================================================

MAX_WIDTH = 1024
MAX_HEIGHT = 832


# ============================================================
# GPU
# ============================================================

print("=" * 65)
print("PHASE 1 EXPERIMENT 2")
print("SDXL Img2Img + Canny ControlNet + IP-Adapter")
print("=" * 65)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 65)


# ============================================================
# CHECK FILES
# ============================================================

for path in [SATELLITE_PATH, STYLE_PATH]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


os.makedirs("data/output", exist_ok=True)


# ============================================================
# LOAD SATELLITE
# ============================================================

print("\nLoading satellite...")

satellite = Image.open(
    SATELLITE_PATH
).convert("RGB")

print("Original satellite:", satellite.size)


# ============================================================
# RESIZE SATELLITE
# ============================================================

scale = min(
    MAX_WIDTH / satellite.width,
    MAX_HEIGHT / satellite.height
)

new_width = int(satellite.width * scale)
new_height = int(satellite.height * scale)

# SDXL-friendly dimensions
new_width = (new_width // 8) * 8
new_height = (new_height // 8) * 8

satellite = satellite.resize(
    (new_width, new_height),
    Image.Resampling.LANCZOS
)

print("Generation size:", satellite.size)


# ============================================================
# LOAD STYLE IMAGE
# ============================================================

print("\nLoading style reference...")

style_image = Image.open(
    STYLE_PATH
).convert("RGB")

print("Style reference:", style_image.size)


# ============================================================
# CREATE CANNY IMAGE
# ============================================================

print("\nCreating Canny control image...")

satellite_np = np.array(satellite)

# Slightly conservative Canny thresholds.
# We don't want every tree texture to become an edge.
gray = cv2.cvtColor(
    satellite_np,
    cv2.COLOR_RGB2GRAY
)

# Small blur reduces tiny satellite texture.
gray = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

low_threshold = 80
high_threshold = 180

edges = cv2.Canny(
    gray,
    low_threshold,
    high_threshold
)

# Convert single-channel edge image to RGB.
edges_rgb = np.stack(
    [edges, edges, edges],
    axis=2
)

canny_image = Image.fromarray(
    edges_rgb
)

canny_image.save(
    CANNY_OUTPUT
)

print("Canny saved:")
print(CANNY_OUTPUT)


# ============================================================
# LOAD IP-ADAPTER IMAGE ENCODER
# ============================================================

print("\nLoading IP-Adapter ViT-H image encoder...")

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    IP_ADAPTER_REPO,
    subfolder="models/image_encoder",
    torch_dtype=torch.float16
)

print("Image encoder loaded.")


# ============================================================
# LOAD CONTROLNET
# ============================================================

print("\nLoading SDXL Canny ControlNet...")

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True
)

print("ControlNet loaded.")


# ============================================================
# LOAD SDXL CONTROLNET IMG2IMG PIPELINE
# ============================================================

print("\nLoading SDXL ControlNet Img2Img...")

pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    SDXL_MODEL,
    controlnet=controlnet,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Important for RTX 4060 8GB.
pipe.enable_model_cpu_offload()

print("SDXL pipeline loaded.")


# ============================================================
# LOAD IP-ADAPTER
# ============================================================

print("\nLoading IP-Adapter Plus...")

pipe.load_ip_adapter(
    IP_ADAPTER_REPO,
    subfolder=IP_ADAPTER_SUBFOLDER,
    weight_name=IP_ADAPTER_WEIGHT
)


# ============================================================
# STYLE-FOCUSED IP-ADAPTER
# ============================================================

# Use the style-focused block rather than all blocks.
#
# This is intended to make the target image behave more like
# a style reference instead of a second geometry source.

style_scale = {
    "up": {
        "block_0": [0.0, 1.0, 0.0]
    }
}

pipe.set_ip_adapter_scale(
    style_scale
)

print("IP-Adapter configured for style-focused conditioning.")


# ============================================================
# PROMPT
# ============================================================

prompt = """
top-down illustrated campus map,
professional landscape architecture map,
clean hand-drawn cartographic illustration,
flat illustrated buildings,
accurate campus site plan,
green lawns,
lush illustrated trees,
clean roads and pathways,
blue roofs,
blue water features,
soft natural colors,
subtle illustrated shadows,
crisp outlines,
detailed architectural site plan,
consistent illustrated map style
"""


negative_prompt = """
photorealistic satellite,
aerial photograph,
3d render,
perspective,
isometric,
text,
labels,
letters,
watermark,
logo,
random architecture,
extra buildings,
missing buildings,
distorted buildings,
warped roads,
curved roads,
destroyed roads,
random structures,
fantasy architecture,
low quality,
blurry,
noise
"""


# ============================================================
# GENERATOR
# ============================================================

generator = torch.Generator(
    device="cpu"
).manual_seed(SEED)


# ============================================================
# GENERATION
# ============================================================

print("\n")
print("=" * 65)
print("GENERATING")
print("=" * 65)

print("Strength:", STRENGTH)
print("ControlNet scale:", CONTROLNET_SCALE)
print("IP-Adapter scale:", IP_ADAPTER_SCALE)
print("Steps:", STEPS)
print("Size:", satellite.size)

print("\nGenerating...")

result = pipe(

    prompt=prompt,

    negative_prompt=negative_prompt,

    # --------------------------------------------------------
    # ACTUAL SATELLITE
    # --------------------------------------------------------

    image=satellite,

    # --------------------------------------------------------
    # CANNY STRUCTURE
    # --------------------------------------------------------

    control_image=canny_image,

    # --------------------------------------------------------
    # STYLE REFERENCE
    # --------------------------------------------------------

    ip_adapter_image=style_image,

    # --------------------------------------------------------
    # DIFFUSION
    # --------------------------------------------------------

    strength=STRENGTH,

    num_inference_steps=STEPS,

    guidance_scale=GUIDANCE,

    # --------------------------------------------------------
    # CONTROLNET
    # --------------------------------------------------------

    controlnet_conditioning_scale=CONTROLNET_SCALE,

    # --------------------------------------------------------
    # RANDOM SEED
    # --------------------------------------------------------

    generator=generator,

).images[0]


# ============================================================
# SAVE
# ============================================================

result.save(
    OUTPUT_PATH
)

print("\n")
print("=" * 65)
print("DONE")
print("=" * 65)

print("Output:")
print(OUTPUT_PATH)

print("Output size:", result.size)

print("\nCanny control image:")
print(CANNY_OUTPUT)