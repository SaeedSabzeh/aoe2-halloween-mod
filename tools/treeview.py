import sys, numpy as np
sys.path.insert(0, ".")
import sld
from PIL import Image

def draw(fr, size=280, bg=(110, 140, 70), dy=40):
    hx, hy = fr["hotspot"]
    out = np.zeros((size, size, 3), np.float64); out[:] = bg
    sh = fr["layers"].get("shadow")
    if sh is not None:
        x1, y1, x2, y2 = sh["rect"]; p = sh["pixels"]
        _blit(out, p[..., None].repeat(3, 2), x1 - hx + size // 2, y1 - hy + size // 2 + dy, lambda o, s: o * (1 - s / 255 * 0.7))
    m = fr["layers"]["main"]
    x1, y1, x2, y2 = m["rect"]; p = m["pixels"]
    _blit(out, p, x1 - hx + size // 2, y1 - hy + size // 2 + dy, None)
    out[size // 2 + dy, size // 2] = (255, 0, 255)
    return out.astype(np.uint8)

def _blit(out, p, ox, oy, fn):
    H, W = out.shape[:2]
    h, w = p.shape[:2]
    X0, Y0 = max(0, ox), max(0, oy)
    X1, Y1 = min(W, ox + w), min(H, oy + h)
    if X0 >= X1 or Y0 >= Y1: return
    src = p[Y0 - oy:Y1 - oy, X0 - ox:X1 - ox].astype(float)
    dst = out[Y0:Y1, X0:X1]
    if fn is not None:
        dst[:] = fn(dst, src)
    else:
        m = src[..., 3] > 127
        dst[m] = src[m][:, :3]

def sheet(frames, path, size=280, cols=6, scale=2):
    tiles = [draw(f, size) for f in frames]
    while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
    img = np.concatenate([np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)], 0)
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(path)
