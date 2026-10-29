import sys, numpy as np
sys.path.insert(0, ".")
import harvest
from PIL import Image

def tile(rgba, sh, hs, T=56):
    out = np.zeros((T, T, 3)); out[:] = (118, 124, 70)
    h, w = rgba.shape[:2]
    ox, oy = T // 2 - hs[0], T - 10 - hs[1]
    for y in range(h):
        for x in range(w):
            X, Y = ox + x, oy + y
            if 0 <= X < T and 0 <= Y < T:
                if sh[y, x]: out[Y, X] *= 1 - sh[y, x] / 255 * 0.7
                if rgba[y, x, 3]: out[Y, X] = rgba[y, x, :3]
    return out.astype(np.uint8)

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    items = [harvest.pumpkin(rng, k) for k in ["jack"] * 6 + ["plain"] * 3 + ["white", "gourd", "gourd"]]
    items += [harvest.hay_bale(rng)[:3] for _ in range(2)] + [harvest.candle(rng) for _ in range(2)] + [harvest.leaves(rng)]
    items += [harvest.skeleton_sitting(rng, 1), harvest.bone_pile(rng)]
    tiles = [tile(*it) for it in items]
    while len(tiles) % 10: tiles.append(np.zeros_like(tiles[0]))
    img = np.concatenate([np.concatenate(tiles[i:i + 10], 1) for i in range(0, len(tiles), 10)], 0)
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save("../check_harvest_props.png")
