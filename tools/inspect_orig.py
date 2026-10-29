import sys, numpy as np
sys.path.insert(0, ".")
import sld
from PIL import Image
import modkit
g = modkit.GAME_GFX

def canvas(fr, size=100):
    # place main layer on a crop of the canvas centred on the hotspot
    hx, hy = fr["hotspot"]
    out = np.zeros((size, size, 3), np.uint8); out[:] = (110, 140, 70)
    for name, col in (("shadow", None), ("main", None)):
        ly = fr["layers"].get(name)
        if ly is None: continue
        x1, y1, x2, y2 = ly["rect"]
        pix = ly["pixels"]
        ox, oy = x1 - hx + size // 2, y1 - hy + size // 2 + 15
        for yy in range(pix.shape[0]):
            for xx in range(pix.shape[1]):
                X, Y = ox + xx, oy + yy
                if 0 <= X < size and 0 <= Y < size:
                    if name == "shadow":
                        out[Y, X] = (out[Y, X] * (1 - pix[yy, xx] / 255 * 0.7)).astype(np.uint8)
                    elif pix[yy, xx, 3] > 127:
                        p = pix[yy, xx, :3]
                        pc = fr["layers"].get("playercolor")
                        if pc is not None and pc["pixels"][yy, xx] > 127:
                            p = np.array([40, 80, 230])
                        out[Y, X] = p
    out[size // 2 + 15, size // 2] = (255, 0, 255)
    return out

def sheet(name, idxs, out, cols=16, size=100):
    _, frames = sld.read(g + name + ".sld")
    tiles = [canvas(frames[i], size) for i in idxs]
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + cols], axis=1) for r in range(0, len(tiles), cols)]
    img = np.concatenate(rows, axis=0)
    Image.fromarray(img).resize((img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(out)

if __name__ == "__main__":
    name, per, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    idxs = [d * per for d in range(16)] + list(range(0, per, max(1, per // 16)))[:16]
    sheet(name, idxs, out, size=int(sys.argv[4]) if len(sys.argv) > 4 else 100)
