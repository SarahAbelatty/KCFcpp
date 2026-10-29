from pathlib import Path

W, H = 320, 240
OUT = Path("test_sequence")
OUT.mkdir(exist_ok=True)

for frame in range(20):
    img = bytearray([35, 35, 35] * (W * H))

    # Static background grid
    for y in range(0, H, 20):
        for x in range(0, W, 20):
            i = (y * W + x) * 3
            img[i:i+3] = bytes([45, 45, 45])

    # Moving, textured object
    ox = 50 + frame * 3
    oy = 80 + frame
    ow, oh = 50, 40

    for y in range(oh):
        for x in range(ow):
            checker = ((x // 5) + (y // 5)) % 2
            if checker:
                color = (240, 180, 30)
            else:
                color = (30, 180, 240)

            # Bright border
            if x < 2 or y < 2 or x >= ow-2 or y >= oh-2:
                color = (255, 255, 255)

            px, py = ox + x, oy + y
            i = (py * W + px) * 3
            img[i:i+3] = bytes(color)

    path = OUT / f"frame{frame:03d}.ppm"
    with path.open("wb") as f:
        f.write(f"P6\n{W} {H}\n255\n".encode())
        f.write(img)

print("Generated 20 test frames.")
