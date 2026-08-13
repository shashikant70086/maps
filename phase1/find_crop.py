from PIL import Image, ImageDraw, ImageFont

INPUT = "data/input/satellite.jpg"
OUTPUT = "data/output/satellite_grid.png"

img = Image.open(INPUT).convert("RGB")
draw = ImageDraw.Draw(img)

W, H = img.size

print("Image size:", W, H)

# Grid spacing
grid = 500

# Draw vertical lines
for x in range(0, W, grid):
    draw.line(
        [(x, 0), (x, H)],
        fill=(255, 0, 0),
        width=3
    )

    draw.text(
        (x + 5, 5),
        f"X={x}",
        fill=(255, 0, 0)
    )

# Draw horizontal lines
for y in range(0, H, grid):
    draw.line(
        [(0, y), (W, y)],
        fill=(255, 0, 0),
        width=3
    )

    draw.text(
        (5, y + 5),
        f"Y={y}",
        fill=(255, 0, 0)
    )

img.save(OUTPUT)

print("Saved:")
print(OUTPUT)