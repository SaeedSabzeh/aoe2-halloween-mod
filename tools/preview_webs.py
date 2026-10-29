import sys, copy, numpy as np
sys.path.insert(0, ".")
import sld, webs, treeview
from PIL import Image
import modkit
g = modkit.GAME_GFX

def webbed(fr, seed, span):
    fr = copy.deepcopy(fr)
    m = fr["layers"]["main"]
    L = webs.web_tree(webs.Layer(m["pixels"], m["rect"]), fr["hotspot"], seed, span)
    m["pixels"], m["rect"] = L.px, L.rect()
    return fr

singles = []
for n, idx in (("n_tree_pine_x1", [0, 1]), ("n_tree_oak_x1", [0, 2]), ("n_tree_jungle_x1", [22, 30])):
    _, f = sld.read(g + n + ".sld")
    for i in idx:
        singles += [f[i], webbed(f[i], i, "jungle" in n)]
treeview.sheet(singles, "../preview_web_trees.png", size=260, cols=6, scale=2)

# mock jungle forest: frames on an iso grid, drawn back to front
_, jf = sld.read(g + "n_tree_jungle_x1.sld")
W, H = 700, 420
out = np.zeros((H, W, 3)); out[:] = (110, 140, 70)
rng = np.random.default_rng(5)
tiles = [(i, j) for i in range(6) for j in range(6)]
tiles.sort(key=lambda t: t[0] + t[1])
for i, j in tiles:
    sx, sy = 350 + (i - j) * 48, 60 + (i + j) * 24
    k = int(rng.integers(21, 39)) if rng.random() < 0.7 else int(rng.integers(0, 21))
    fr = webbed(jf[k], i * 7 + j, True)
    hx, hy = fr["hotspot"]
    for name in ("shadow", "main"):
        ly = fr["layers"].get(name)
        if ly is None: continue
        x1, y1, _, _ = ly["rect"]
        px = ly["pixels"]
        if name == "shadow":
            treeview._blit(out, px[..., None].repeat(3, 2), sx + x1 - hx, sy + y1 - hy, lambda o, s: o * (1 - s / 255 * 0.6))
        else:
            treeview._blit(out, px, sx + x1 - hx, sy + y1 - hy, None)
Image.fromarray(out.astype(np.uint8)).resize((W * 2, H * 2), Image.NEAREST).save("../preview_web_jungle.png")
