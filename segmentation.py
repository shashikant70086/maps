import os
import torch
import numpy as np

from PIL import Image

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Pranilllllll/segformer-satellite-segementation"

INPUT_IMAGE = "data/input/satellite.jpg"

OUTPUT_MASK = "data/output/segmentation_mask.png"

OUTPUT_COLOR = "data/output/segmentation_color.png"


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("SEGMENTATION")
print("=" * 60)

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(INPUT_IMAGE).convert("RGB")

print("Original image size:", image.size)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading SegFormer...")

processor = SegformerImageProcessor.from_pretrained(
    MODEL_NAME
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
model.eval()

print("SegFormer loaded.")


# ============================================================
# PREPROCESS
# ============================================================

print("\nPreprocessing image...")

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ============================================================
# INFERENCE
# ============================================================

print("Running segmentation...")

with torch.no_grad():

    outputs = model(**inputs)

logits = outputs.logits


# ============================================================
# RESIZE MASK TO ORIGINAL IMAGE
# ============================================================

print("Resizing segmentation...")

logits = torch.nn.functional.interpolate(
    logits,
    size=(image.height, image.width),
    mode="bilinear",
    align_corners=False,
)

prediction = logits.argmax(
    dim=1
)[0].cpu().numpy()


# ============================================================
# SAVE RAW CLASS MASK
# ============================================================

os.makedirs("data/output", exist_ok=True)

mask = Image.fromarray(
    prediction.astype(np.uint8)
)

mask.save(OUTPUT_MASK)


# ============================================================
# COLOR MAP
# ============================================================

# Classes published by the model:
#
# 0 = background
# 1 = residential_area
# 2 = road
# 3 = river
# 4 = forest
# 5 = unused_land
# 6 = agricultural_area

colors = np.array(
    [
        [0,   0,   0],       # background
        [220, 80,  80],      # residential
        [80,  80,  80],      # road
        [50,  100, 220],     # river
        [30, 130, 60],       # forest
        [180, 160, 80],      # unused
        [130, 200, 80],      # agriculture
    ],
    dtype=np.uint8
)


color_mask = colors[prediction]

color_image = Image.fromarray(color_mask)

color_image.save(OUTPUT_COLOR)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("Raw mask:")
print(os.path.abspath(OUTPUT_MASK))

print("\nColor segmentation:")
print(os.path.abspath(OUTPUT_COLOR))