"""Mod thumbnail (1280x720) composed from the mod's own sprites."""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
import sld
import modkit

MOD = os.path.join(modkit.MODS_LOCAL, "Halloween")
ANIMALS_MOD = os.path.join(modkit.MODS_LOCAL, "Halloween Animals")
GFX = os.path.join(MOD, "resources", "_common", "drs", "graphics") + "/"
TEX = modkit.GAME_DIR + "resources/_common/terrain/textures/2x/"
W, H = 640, 360          # composed at game scale, then doubled
BLUE = np.array([46, 96, 226])


def frame(name, k=0):
    return sld.read(GFX + name + ".sld")[1][k]


def blit(img, fr, x, y):
    hx, hy = fr["hotspot"]
    sh = fr["layers"].get("shadow")
    if sh is not None:
        sx1, sy1, _, _ = sh["rect"]
        p = sh["pixels"].astype(float) / 255
        _over(img, p, x + sx1 - hx, y + sy1 - hy, lambda dst, s: dst * (1 - 0.65 * s[..., None]))
    m = fr["layers"].get("main")
    if m is None:
        return
    x1, y1, _, _ = m["rect"]
    px = m["pixels"].astype(float)
    pc = fr["layers"].get("playercolor")
    if pc is not None:
        t = pc["pixels"].astype(float)[..., None] / 255
        lum = px[..., :3].mean(-1, keepdims=True) / 160
        px[..., :3] = px[..., :3] * (1 - t) + np.clip(BLUE * lum, 0, 255) * t
    _over(img, px, x + x1 - hx, y + y1 - hy, None)


def _over(img, src, ox, oy, fn):
    h, w = src.shape[:2]
    X0, Y0, X1, Y1 = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
    if X0 >= X1 or Y0 >= Y1:
        return
    s = src[Y0 - oy:Y1 - oy, X0 - ox:X1 - ox]
    d = img[Y0:Y1, X0:X1]
    if fn is not None:
        d[:] = fn(d, s)
    else:
        m = s[..., 3] > 127
        d[m] = s[m][:, :3]


def main(out_path, animals=False):
    """animals: thumbnail for the animals-only mod (game's own trees and Town Center)."""
    global GFX
    if animals:
        GFX = os.path.join(ANIMALS_MOD, "resources", "_common", "drs", "graphics") + "/"
    scenery = modkit.GAME_GFX if animals else GFX
    grass = Image.open(TEX + "g_gr2.dds").convert("RGB").resize((1024, 1024))
    img = np.asarray(grass, dtype=float)[:H, :W].copy()
    # dusk grade: cooler, darker towards the top, warm glow low in the frame
    yy = np.linspace(0, 1, H)[:, None, None]
    img = img * (0.55 + 0.35 * yy) * np.array([0.92, 0.9, 1.08])

    items = []
    # haunted forest on the left (grid of webbed pines)
    tree = sld.read(scenery + "n_tree_oak_x1.sld")[1]
    for i in range(4):
        for j in range(4):
            items.append((40 + (i - j) * 48 + 120, 70 + (i + j) * 24, tree[(i * 5 + j * 3) % len(tree)]))
    items.append((360, 205, sld.read(scenery + "b_west_town_center_age2_x1.sld")[1][0]))
    spider = sld.read(GFX + "a_herd_sheep_walkA_x1.sld")[1]
    for k, (x, y, d) in enumerate(((300, 300, 2), (345, 318, 1), (392, 296, 3), (258, 276, 14))):
        items.append((x, y, spider[d * 30 + k * 7]))
    deer = sld.read(GFX + "a_hunt_deer_runA_x1.sld")[1]
    items += [(560, 170, deer[10 * 30 + 4]), (600, 214, deer[10 * 30 + 15])]
    boar = sld.read(GFX + "a_hunt_boar_attackA_x1.sld")[1]
    items.append((520, 300, boar[6 * 45 + 20]))
    if animals:
        items += [(190, 318, spider[5 * 30 + 3]), (455, 340, boar[3 * 45 + 8]), (610, 300, deer[12 * 30 + 9])]
    else:
        items.append((150, 320, frame("n_forage_bush_x1", 1)))
    for x, y, fr in sorted(items, key=lambda t: t[1]):
        blit(img, fr, x, y)

    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).resize((W * 2, H * 2), Image.NEAREST)
    # vignette
    v = Image.new("L", out.size, 0)
    ImageDraw.Draw(v).ellipse([-200, -160, out.size[0] + 200, out.size[1] + 260], fill=255)
    v = v.filter(ImageFilter.GaussianBlur(120))
    out = Image.composite(out, Image.new("RGB", out.size, (10, 6, 16)), v)

    d = ImageDraw.Draw(out)
    font = lambda name, size: ImageFont.truetype("C:/Windows/Fonts/" + name, size)
    try:
        title, sub = font("impact.ttf", 118), font("georgiab.ttf", 34)
    except OSError:
        title = sub = ImageFont.load_default()
    lines = ((("HALLOWEEN ANIMALS", "Spider Sheep · Ghosts · Skeleton Boars · Ghost Fish") if animals else
              ("HALLOWEEN", "Spiders · Ghosts · Haunted Woods · Pumpkin Farms")))
    for text, f, y, fill in ((lines[0], title, 22, (255, 150, 34)), (lines[1], sub, 160, (236, 226, 210))):
        tw = d.textlength(text, font=f)
        x = (out.size[0] - tw) / 2
        d.text((x, y), text, font=f, fill=fill, stroke_width=6 if f is title else 3, stroke_fill=(20, 8, 4))
    out.save(out_path, quality=92)


if __name__ == "__main__":
    if "--animals" in sys.argv:
        main(os.path.join(ANIMALS_MOD, "thumbnail.jpg"), animals=True)
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(MOD, "thumbnail.jpg"))
