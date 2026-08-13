from PIL import Image, ImageDraw
import os


SATELLITE = "data/input/satellite.jpg"
TARGET = "data/reference/target.png"

OUTPUT = "data/output/alignment_overview.png"


# --------------------------------------------------
# Load
# --------------------------------------------------

satellite = Image.open(SATELLITE).convert("RGB")
target = Image.open(TARGET).convert("RGB")

print("Satellite:", satellite.size)
print("Target:", target.size)


# --------------------------------------------------
# Resize while preserving aspect ratio
# --------------------------------------------------

def resize_to_width(image, width):
    ratio = width / image.width
    height = int(image.height * ratio)
    return image.resize((width, height))


display_width = 900

sat_display = resize_to_width(
    satellite,
    display_width
)

target_display = resize_to_width(
    target,
    display_width
)


# --------------------------------------------------
# Create canvas
# --------------------------------------------------

margin = 30
label_height = 60

canvas_width = display_width + 2 * margin

canvas_height = (
    label_height
    + sat_display.height
    + label_height
    + target_display.height
    + 3 * margin
)

canvas = Image.new(
    "RGB",
    (canvas_width, canvas_height),
    "white"
)

draw = ImageDraw.Draw(canvas)


# --------------------------------------------------
# Satellite
# --------------------------------------------------

y = margin

draw.text(
    (margin, y),
    "SATELLITE IMAGE",
    fill="black"
)

y += label_height

canvas.paste(
    sat_display,
    (margin, y)
)

y += sat_display.height + margin


# --------------------------------------------------
# Target
# --------------------------------------------------

draw.text(
    (margin, y),
    "TARGET MAP",
    fill="black"
)

y += label_height

canvas.paste(
    target_display,
    (margin, y)
)


# --------------------------------------------------
# Save
# --------------------------------------------------

os.makedirs(
    "data/output",
    exist_ok=True
)

canvas.save(OUTPUT)

print()
print("Saved:")
print(OUTPUT)