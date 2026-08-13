from PIL import Image
import os

INPUT = "data/input/satellite.jpg"
OUTPUT = "data/output/highres_test_crop.png"

image = Image.open(INPUT).convert("RGB")

print("Original:", image.size)

# Central campus region.
# We will adjust these coordinates if necessary.
left = 1800
top = 1500
right = 3336
bottom = 3036

crop = image.crop(
    (left, top, right, bottom)
)

print("Crop:", crop.size)

os.makedirs("data/output", exist_ok=True)

crop.save(OUTPUT)

print("Saved:")
print(OUTPUT)