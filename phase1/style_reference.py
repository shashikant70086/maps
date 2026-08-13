from PIL import Image, ImageOps
import os

TARGET = "data/reference/target.png"
OUTPUT = "data/output/style_reference.png"

os.makedirs("data/output", exist_ok=True)

image = Image.open(TARGET).convert("RGB")

print("Original target:", image.size)

# Create a square style-reference canvas.
# We preserve the whole target instead of cropping away important
# visual characteristics.
canvas_size = 1024

style_image = ImageOps.contain(
    image,
    (canvas_size, canvas_size)
)

canvas = Image.new(
    "RGB",
    (canvas_size, canvas_size),
    (240, 240, 240)
)

x = (canvas_size - style_image.width) // 2
y = (canvas_size - style_image.height) // 2

canvas.paste(style_image, (x, y))

canvas.save(OUTPUT)

print("Style reference saved to:")
print(OUTPUT)
print("Size:", canvas.size)