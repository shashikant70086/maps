import os
import cv2
import torch

from PIL import Image

from transformers import CLIPVisionModelWithProjection

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

CONTROLNET_MODEL = "diffusers/controlnet-canny-sdxl-1.0"

IP_ADAPTER_REPO = "h94/IP-Adapter"

IMAGE_ENCODER_SUBFOLDER = "models/image_encoder"

IP_ADAPTER_SUBFOLDER = "sdxl_models"

IP_ADAPTER_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"


INPUT_IMAGE = "data/input/satellite.jpg"

REFERENCE_IMAGE = "data/reference/target.png"

OUTPUT_IMAGE = "data/output/generated_map_controlnet.png"


# ============================================================
# GPU
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available.")

device = "cuda"
dtype = torch.float16

print("=" * 60)
print("GPU")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(INPUT_IMAGE):
    raise FileNotFoundError(INPUT_IMAGE)

if not os.path.exists(REFERENCE_IMAGE):
    raise FileNotFoundError(REFERENCE_IMAGE)

os.makedirs("data/output", exist_ok=True)


# ============================================================
# LOAD SATELLITE
# ============================================================

print("\nLoading satellite image...")

satellite = Image.open(
    INPUT_IMAGE
).convert("RGB")

print("Satellite size:", satellite.size)


# ============================================================
# CREATE CANNY IMAGE
# ============================================================

print("\nCreating Canny control image...")

# Convert PIL → OpenCV
satellite_cv = cv2.cvtColor(
    __import__("numpy").array(satellite),
    cv2.COLOR_RGB2GRAY
)

# Slight blur to remove satellite noise
blur = cv2.GaussianBlur(
    satellite_cv,
    (5, 5),
    0
)

# Detect edges
edges = cv2.Canny(
    blur,
    threshold1=80,
    threshold2=180
)

# Convert back to RGB PIL image
canny_image = Image.fromarray(edges).convert("RGB")

print("Canny image created.")


# ============================================================
# LOAD IP-ADAPTER IMAGE ENCODER
# ============================================================

print("\nLoading IP-Adapter image encoder...")

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    IP_ADAPTER_REPO,
    subfolder=IMAGE_ENCODER_SUBFOLDER,
    torch_dtype=dtype,
)

print("Image encoder loaded.")


# ============================================================
# LOAD CONTROLNET
# ============================================================

print("\nLoading ControlNet Canny...")

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=dtype,
)

print("ControlNet loaded.")


# ============================================================
# LOAD SDXL + CONTROLNET
# ============================================================

print("\nLoading SDXL + ControlNet...")

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    image_encoder=image_encoder,
    torch_dtype=dtype,
)

pipe = pipe.to(device)

print("SDXL + ControlNet loaded.")


# ============================================================
# LOAD IP-ADAPTER
# ============================================================

print("\nLoading IP-Adapter Plus...")

pipe.load_ip_adapter(
    IP_ADAPTER_REPO,
    subfolder=IP_ADAPTER_SUBFOLDER,
    weight_name=IP_ADAPTER_WEIGHT,
)

pipe.set_ip_adapter_scale(0.7)

print("IP-Adapter loaded.")


# ============================================================
# REFERENCE IMAGE
# ============================================================

reference = Image.open(
    REFERENCE_IMAGE
).convert("RGB")

print("\nReference size:", reference.size)


# ============================================================
# PROMPT
# ============================================================

prompt = """
top-down orthographic illustrated campus masterplan,
professional landscape architecture illustration,
accurate campus layout,
clean simplified buildings,
illustrated individual trees,
lush green grass,
clean gray roads,
blue roofs and water,
white and light gray buildings,
subtle shadows,
clean geometric shapes,
professional architectural site plan,
detailed illustrated campus map,
consistent hand illustrated map style
"""


negative_prompt = """
photorealistic,
satellite photograph,
aerial photograph,
perspective view,
3d render,
people,
cars,
text,
letters,
labels,
watermark,
logo,
blurry,
low quality,
distorted buildings,
random buildings,
random roads,
fantasy architecture
"""


# ============================================================
# GENERATION
# ============================================================

print("\nGenerating map...")

generator = torch.Generator(
    device="cuda"
).manual_seed(42)


with torch.inference_mode():

    result = pipe(
        prompt=prompt,

        negative_prompt=negative_prompt,

        image=canny_image,

        ip_adapter_image=reference,

        controlnet_conditioning_scale=0.8,

        width=768,
        height=768,

        num_inference_steps=25,

        guidance_scale=5.0,

        generator=generator,
    ).images[0]


# ============================================================
# SAVE
# ============================================================

result.save(OUTPUT_IMAGE)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("Output:")
print(os.path.abspath(OUTPUT_IMAGE))