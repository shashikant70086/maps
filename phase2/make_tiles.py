from PIL import Image, ImageDraw
import os
import math

INPUT = "data/input/satellite.jpg"

TILE_SIZE = 1536
OVERLAP = 384
STRIDE = TILE_SIZE - OVERLAP

OUTPUT_DIR = "data/tiles"

os.makedirs(OUTPUT_DIR, exist_ok=True)

image = Image.open(INPUT).convert("RGB")

W, H = image.size

print("Source:", W, "x", H)
print("Tile:", TILE_SIZE)
print("Overlap:", OVERLAP)
print("Stride:", STRIDE)

tiles = []

tile_id = 0

y = 0

while y < H:

    x = 0

    while x < W:

        # Make sure final tile reaches image boundary
        x1 = min(x + TILE_SIZE, W)
        y1 = min(y + TILE_SIZE, H)

        x0 = x1 - TILE_SIZE
        y0 = y1 - TILE_SIZE

        # Prevent negative coordinates
        x0 = max(0, x0)
        y0 = max(0, y0)

        crop = image.crop(
            (x0, y0, x1, y1)
        )

        filename = (
            f"tile_{tile_id:03d}_"
            f"x{x0}_y{y0}.png"
        )

        path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        crop.save(path)

        tiles.append({
            "id": tile_id,
            "x": x0,
            "y": y0,
            "path": path
        })

        print(
            f"Tile {tile_id:03d}: "
            f"x={x0}, y={y0}, "
            f"size={crop.size}"
        )

        tile_id += 1

        if x + TILE_SIZE >= W:
            break

        x += STRIDE

    if y + TILE_SIZE >= H:
        break

    y += STRIDE


print()
print("=" * 50)
print("DONE")
print("=" * 50)

print("Total tiles:", len(tiles))