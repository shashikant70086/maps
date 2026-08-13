import os
import re
import cv2
import torch
import numpy as np

from PIL import Image
from transformers import CLIPVisionModelWithProjection

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
)


# ============================================================
# PATHS
# ============================================================

TILES_DIR = "data/tiles"
OUTPUT_DIR = "data/generated_tiles"
CANNY_DIR = "data/canny_tiles"

STYLE_IMAGE_PATH = "data/reference/target.png"


# ============================================================
# MODELS
# ============================================================

SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

CONTROLNET_MODEL = (
    "diffusers/controlnet-canny-sdxl-1.0"
)

IP_ADAPTER_REPO = "h94/IP-Adapter"

IP_ADAPTER_SUBFOLDER = "sdxl_models"

IP_ADAPTER_WEIGHT = (
    "ip-adapter-plus_sdxl_vit-h.safetensors"
)


# ============================================================
# GENERATION SETTINGS
# ============================================================

STRENGTH = 0.20

CONTROLNET_SCALE = 0.65

IP_ADAPTER_SCALE = 0.90

STEPS = 30

GUIDANCE = 5.0

SEED = 42


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("PHASE 2 — TILE GENERATION")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 70)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CANNY_DIR, exist_ok=True)


# ============================================================
# CHECK STYLE IMAGE
# ============================================================

if not os.path.exists(STYLE_IMAGE_PATH):

    raise FileNotFoundError(
        f"\nStyle reference not found:\n"
        f"{STYLE_IMAGE_PATH}"
    )


style_image = Image.open(
    STYLE_IMAGE_PATH
).convert("RGB")

print(
    "\nStyle reference:",
    style_image.size
)


# ============================================================
# FIND TILES
# ============================================================

tile_files = [
    f
    for f in os.listdir(TILES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
    and f.lower().startswith("tile_")
]


def tile_sort_key(filename):

    match = re.search(
        r"tile_(\d+)",
        filename
    )

    if match:
        return int(match.group(1))

    return 999999


tile_files.sort(
    key=tile_sort_key
)


if not tile_files:

    raise FileNotFoundError(
        f"\nNo tiles found in:\n{TILES_DIR}"
    )


print(
    f"\nFound {len(tile_files)} tiles."
)


for i, filename in enumerate(tile_files):

    print(
        f"  [{i + 1:02d}/{len(tile_files):02d}] "
        f"{filename}"
    )


# ============================================================
# LOAD IP-ADAPTER IMAGE ENCODER
# ============================================================

print("\n")
print("Loading IP-Adapter ViT-H image encoder...")

image_encoder = (
    CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER_REPO,
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    )
)

print("Image encoder loaded.")


# ============================================================
# LOAD CONTROLNET
# ============================================================

print("\nLoading ControlNet Canny SDXL...")

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET_MODEL,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

print("ControlNet loaded.")


# ============================================================
# LOAD SDXL
# ============================================================

print("\nLoading SDXL...")

pipe = (
    StableDiffusionXLControlNetImg2ImgPipeline
    .from_pretrained(
        SDXL_MODEL,
        controlnet=controlnet,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
)

# Important for RTX 4060 8 GB
pipe.enable_model_cpu_offload()

print("SDXL loaded.")


# ============================================================
# LOAD IP-ADAPTER
# ============================================================

print("\nLoading IP-Adapter Plus...")

pipe.load_ip_adapter(
    IP_ADAPTER_REPO,
    subfolder=IP_ADAPTER_SUBFOLDER,
    weight_name=IP_ADAPTER_WEIGHT,
)

pipe.set_ip_adapter_scale(
    IP_ADAPTER_SCALE
)

print("IP-Adapter loaded.")


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
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


NEGATIVE_PROMPT = """
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
random structures,
fantasy architecture,
low quality,
blurry,
noise
"""


# ============================================================
# CANNY GENERATION
# ============================================================

def create_canny(tile):

    tile_np = np.array(tile)

    gray = cv2.cvtColor(
        tile_np,
        cv2.COLOR_RGB2GRAY
    )

    # Reduce tiny satellite texture.
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        80,
        180
    )

    edges_rgb = np.stack(
        [edges, edges, edges],
        axis=2
    )

    return Image.fromarray(
        edges_rgb
    )


# ============================================================
# PROCESS TILES
# ============================================================

total = len(tile_files)


for index, tile_filename in enumerate(
    tile_files,
    start=1
):

    tile_path = os.path.join(
        TILES_DIR,
        tile_filename
    )

    # --------------------------------------------------------
    # Output name
    # --------------------------------------------------------

    output_filename = (
        os.path.splitext(tile_filename)[0]
        + "_generated.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    canny_filename = (
        os.path.splitext(tile_filename)[0]
        + "_canny.png"
    )

    canny_path = os.path.join(
        CANNY_DIR,
        canny_filename
    )


    # --------------------------------------------------------
    # SKIP IF ALREADY DONE
    # --------------------------------------------------------

    if os.path.exists(output_path):

        print("\n")
        print("-" * 70)

        print(
            f"[{index}/{total}] "
            f"Already exists — SKIPPING"
        )

        print(output_path)

        continue


    # --------------------------------------------------------
    # LOAD TILE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)

    print(
        f"[{index}/{total}] "
        f"PROCESSING TILE"
    )

    print(
        tile_filename
    )

    print("=" * 70)


    tile = Image.open(
        tile_path
    ).convert("RGB")


    print(
        "Tile size:",
        tile.size
    )


    # --------------------------------------------------------
    # CREATE CANNY
    # --------------------------------------------------------

    print(
        "Creating Canny..."
    )

    canny_image = create_canny(
        tile
    )

    canny_image.save(
        canny_path
    )

    print(
        "Canny saved:",
        canny_path
    )


    # --------------------------------------------------------
    # GENERATOR
    # --------------------------------------------------------

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(SEED)


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print(
        "\nGenerating..."
    )

    result = pipe(

        prompt=PROMPT,

        negative_prompt=NEGATIVE_PROMPT,

        # Original high-resolution tile
        image=tile,

        # Structural information
        control_image=canny_image,

        # Same style reference for every tile
        ip_adapter_image=style_image,

        strength=STRENGTH,

        num_inference_steps=STEPS,

        guidance_scale=GUIDANCE,

        controlnet_conditioning_scale=(
            CONTROLNET_SCALE
        ),

        generator=generator,

    ).images[0]


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result.save(
        output_path
    )


    print("\n")
    print(
        "Tile generated successfully."
    )

    print(
        "Output:",
        output_path
    )


    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{allocated:.2f} GB allocated / "
            f"{reserved:.2f} GB reserved"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("ALL TILE PROCESSING COMPLETE")
print("=" * 70)

print(
    "Generated tiles:",
    OUTPUT_DIR
)

print(
    "Canny tiles:",
    CANNY_DIR
)

print(
    "\nNext step:"
)

print(
    "Run stitch_tiles.py"
)