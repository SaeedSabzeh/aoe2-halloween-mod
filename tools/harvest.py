"""Autumn harvest props for Town Centers, lit to sit in with the game's art.

Each prop renderer returns (rgba, shadow, hotspot) with the hotspot on the
ground under the prop. Every call can produce a different prop from its rng.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import render3d
import poly3d
from render3d import Scene

ORANGES = [(206, 104, 26), (190, 88, 22), (220, 124, 38), (176, 80, 24), (212, 112, 30)]
PALE = [(222, 214, 188), (200, 196, 160)]            # white pumpkins
GOURDS = [(150, 150, 64), (196, 168, 70), (118, 128, 58)]
STEMS = [(98, 86, 44), (84, 92, 46), (110, 92, 52)]
GLOW_CORE = np.array([255, 228, 140])
GLOW_EDGE = np.array([236, 132, 34])
CUT_RIM = np.array([104, 44, 14])


def _soft(scene, facing, size, ground, seed):
    return render3d.render(scene, facing, size, ground=ground, outline=None, soft=True,
                           noise=0.05, seed=seed, shadow_alpha=110)


# ------------------------------------------------------------------ pumpkins

def _pumpkin_scene(rng, R, tall, color, stem):
    s = Scene()
    lobes = int(rng.integers(7, 11))
    H = R * tall
    s.sphere((0, 0, H * 0.62), R * 0.78 * min(1.0, tall + 0.1), np.array(color) * 0.82)
    for k in range(lobes):
        a = k * 2 * np.pi / lobes + rng.uniform(-0.08, 0.08)
        c = np.clip(np.array(color) * rng.uniform(0.9, 1.06), 0, 255)
        x, y = np.cos(a) * R * 0.5, np.sin(a) * R * 0.5
        s.capsule((x, y, H * 0.5), (x, y, H * 0.78), R * 0.5, c)
    # curled stem
    lean = rng.uniform(-1, 1, 2) * R * 0.25
    top = H * 1.18
    s.capsule((0, 0, top - R * 0.15), (lean[0], lean[1], top + R * 0.3), R * 0.13, stem, r1=R * 0.1)
    return s


def _carve(rgba, hotspot, R, tall, rng):
    """Cut a face into the front of the pumpkin, with a warm glow inside the holes."""
    h, w = rgba.shape[:2]
    S = 4
    mask = Image.new("L", (w * S, h * S), 0)
    d = ImageDraw.Draw(mask)
    hx, hy = hotspot
    cx = hx * S
    cy = (hy - R * tall * 0.62 * np.cos(np.radians(30))) * S   # pumpkin centre on screen
    u = R * S
    style = rng.integers(4)
    eye_y = -0.28
    for side in (-1, 1):
        ex = side * 0.38
        if style == 0:      # classic triangles
            d.polygon([(cx + (ex - 0.17) * u, cy + (eye_y + 0.1) * u), (cx + (ex + 0.17) * u, cy + (eye_y + 0.1) * u),
                       (cx + ex * u, cy + (eye_y - 0.2) * u)], fill=255)
        elif style == 1:    # round
            d.ellipse([cx + (ex - 0.13) * u, cy + (eye_y - 0.13) * u, cx + (ex + 0.13) * u, cy + (eye_y + 0.11) * u], fill=255)
        elif style == 2:    # angry slants
            d.polygon([(cx + (ex - 0.18 * side) * u, cy + (eye_y - 0.12) * u), (cx + (ex + 0.18 * side) * u, cy + (eye_y + 0.02) * u),
                       (cx + (ex - 0.1 * side) * u, cy + (eye_y + 0.12) * u)], fill=255)
        else:               # crescents
            d.chord([cx + (ex - 0.16) * u, cy + (eye_y - 0.16) * u, cx + (ex + 0.16) * u, cy + (eye_y + 0.12) * u], 200, 340, fill=255)
    if style != 1 and rng.random() < 0.6:   # nose
        d.polygon([(cx - 0.07 * u, cy - 0.02 * u), (cx + 0.07 * u, cy - 0.02 * u), (cx, cy - 0.15 * u)], fill=255)
    mouth_w = rng.uniform(0.45, 0.62)
    if style == 1 and rng.random() < 0.5:   # surprised O
        d.ellipse([cx - 0.12 * u, cy + 0.08 * u, cx + 0.12 * u, cy + 0.34 * u], fill=255)
    else:
        teeth = int(rng.integers(2, 5))
        top = [(cx - mouth_w * u, cy + 0.08 * u)]
        for i in range(teeth * 2 + 1):
            f = -mouth_w + (i + 1) * 2 * mouth_w / (teeth * 2 + 2)
            top.append((cx + f * u, cy + (0.08 + (0.1 if i % 2 == 0 else 0.0)) * u))
        top.append((cx + mouth_w * u, cy + 0.08 * u))
        bottom = [(cx + mouth_w * 0.6 * u, cy + 0.32 * u), (cx, cy + 0.4 * u), (cx - mouth_w * 0.6 * u, cy + 0.32 * u)]
        d.polygon(top + bottom, fill=255)
    m_hi = np.asarray(mask, dtype=np.float64) / 255
    glow = np.asarray(mask.filter(ImageFilter.GaussianBlur(S * 1.2)), dtype=np.float64) / 255
    m = np.asarray(mask.resize((w, h), Image.BOX), dtype=np.float64) / 255
    g = np.asarray(Image.fromarray((glow * 255).astype(np.uint8)).resize((w, h), Image.BOX), dtype=np.float64) / 255
    # the upper edge of each hole shows the cut flesh, darker than the glow below it
    rim_hi = np.clip(m_hi - np.roll(m_hi, S, 0), 0, 1)
    rim = np.asarray(Image.fromarray((rim_hi * 255).astype(np.uint8)).resize((w, h), Image.BOX), dtype=np.float64) / 255
    hole = (m > 0.45) & (rgba[..., 3] > 0)
    t = np.clip(g / 0.5, 0, 1)[..., None]
    col = GLOW_EDGE * (1 - t) + GLOW_CORE * t
    col = col * (1 - rim[..., None] * 0.8) + CUT_RIM * rim[..., None] * 0.8
    rgba[hole, :3] = np.clip(col[hole], 0, 255)
    # faint warm spill around the holes onto the skin
    spill = (g > 0.15) & ~hole & (rgba[..., 3] > 0)
    rgba[spill, :3] = np.clip(rgba[spill, :3] * (1 + 0.45 * g[spill, None]), 0, 255)
    return rgba


def pumpkin(rng, kind="jack", R=None):
    """kind: jack (carved, glowing), plain, white, gourd."""
    if kind == "gourd":
        R = R or rng.uniform(2.6, 3.6)
        tall = rng.uniform(1.0, 1.5)
        color = GOURDS[rng.integers(len(GOURDS))]
    else:
        R = R or (rng.uniform(6.5, 10.0) if kind == "jack" else rng.uniform(4.5, 8.5))
        tall = rng.uniform(0.68, 1.08)
        color = (PALE if kind == "white" else ORANGES)[rng.integers(len(PALE) if kind == "white" else len(ORANGES))]
    scene = _pumpkin_scene(rng, R, tall, color, STEMS[rng.integers(len(STEMS))])
    size = int(R * 2.6) + 8
    size += (-size) % 4
    rgba, _, sh, hs = _soft(scene, rng.uniform(0, 6.28), (size, size + 4), 4, int(rng.integers(1 << 30)))
    if kind == "jack":
        rgba = _carve(rgba, hs, R, tall, rng)
    return rgba, sh, hs


# ------------------------------------------------------------------ hay bales, candles, leaves

def hay_bale(rng):
    s = poly3d.Scene()
    w, d, h = rng.uniform(15, 19), rng.uniform(10, 12), rng.uniform(8, 10)
    s.box((w, d, h), (206, 176, 96), top=(222, 196, 116), side_color=(188, 158, 84))
    facing = rng.uniform(-0.5, 0.5) + (np.pi / 2 if rng.random() < 0.5 else 0)
    rgba, _, sh, hs = poly3d.render(s, facing, (40, 36), ground=8, outline=None, shadow_alpha=110)
    # straw: streaks and twine
    hh, ww = rgba.shape[:2]
    streak = np.random.default_rng(int(rng.integers(1 << 30))).normal(0, 1, (hh, ww))
    streak = np.cumsum(streak, axis=1) * 0.08
    streak = streak - streak.mean(axis=1, keepdims=True)
    op = rgba[..., 3] > 0
    grain = np.random.default_rng(int(rng.integers(1 << 30))).normal(0, 0.09, (hh, ww))
    rgba[op, :3] = np.clip(rgba[op, :3] * (1 + 0.22 * np.tanh(streak[op]) + grain[op])[:, None], 0, 255)
    # loose straw sticking out along the silhouette
    edge = op & ~(np.roll(op, 1, 0) & np.roll(op, -1, 0) & np.roll(op, 1, 1) & np.roll(op, -1, 1))
    ey, ex = np.nonzero(edge)
    for i in np.random.default_rng(int(rng.integers(1 << 30))).choice(len(ey), size=min(len(ey), 14), replace=False):
        y, x = ey[i] + rng.integers(-1, 2), ex[i] + rng.integers(-1, 2)
        if 0 <= y < hh and 0 <= x < ww:
            rgba[y, x] = (214, 186, 108, 255)
    top_pts = np.nonzero(op)
    top_y = int(top_pts[0].min()) if len(top_pts[0]) else hs[1]
    return rgba, sh, hs, hs[1] - top_y - 2


def candle(rng):
    s = Scene()
    hgt = rng.uniform(4.5, 7.5)
    s.capsule((0, 0, 1.2), (0, 0, hgt), 1.3, (232, 222, 196))
    rgba, _, sh, hs = _soft(s, 0, (12, 16), 3, int(rng.integers(1 << 30)))
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    if len(ys):
        fy, fx = ys.min() - 2, int(xs.mean())
        if fy >= 2:
            rgba[fy:fy + 3, fx] = (255, 196, 90, 255)
            rgba[fy - 1:fy + 1, fx] = (255, 240, 180, 255)
            rgba[fy + 1, fx - 1] = rgba[fy + 1, fx + 1] = (232, 140, 50, 255)
    return rgba, sh, hs


def leaves(rng, w=44, h=24, density=1.0):
    """Pumpkin vines and leaves lying on the ground."""
    S = 4
    img = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy = w * S / 2, h * S / 2
    for _ in range(int(rng.integers(2, 4))):   # vines
        pts, x, y = [], cx + rng.uniform(-w, w) * S * 0.3, cy + rng.uniform(-h, h) * S * 0.25
        ang = rng.uniform(0, 6.28)
        for _ in range(14):
            pts.append((x, y))
            ang += rng.uniform(-0.6, 0.6)
            x += np.cos(ang) * S * 2.2
            y += np.sin(ang) * S * 1.1
        d.line(pts, fill=(86, 96, 44, 255), width=S)
    for _ in range(int(rng.integers(7, 12) * density)):  # leaves
        x, y = cx + rng.uniform(-0.42, 0.42) * w * S, cy + rng.uniform(-0.35, 0.35) * h * S
        r = rng.uniform(2.2, 3.6) * S * (1 + 0.25 * (density > 1))
        g = [(58, 88, 36), (72, 104, 44), (50, 76, 34), (96, 104, 46)][rng.integers(4)]
        d.ellipse([x - r, y - r * 0.55, x + r, y + r * 0.55], fill=g + (255,))
        d.line([(x - r * 0.7, y), (x + r * 0.7, y)], fill=tuple(int(v * 0.75) for v in g) + (255,), width=2)
    a = np.asarray(img.resize((w, h), Image.BOX)).copy()
    a[..., 3] = np.where(a[..., 3] > 110, 255, 0)
    return a, np.zeros((h, w), np.uint8), (w // 2, h // 2)


def skeleton_sitting(rng, mirror):
    import skeleton
    scene = skeleton.sitting(mirror)
    facing = -0.4 if mirror > 0 else np.pi + 0.4
    rgba, _, sh, hs = _soft(scene, facing, (52, 48), 12, int(rng.integers(1 << 30)))
    return rgba, sh, hs


def bone_pile(rng):
    import skeleton
    rgba, _, sh, hs = _soft(skeleton.bone_pile(int(rng.integers(100))), rng.uniform(0, 6.28), (52, 48), 12,
                            int(rng.integers(1 << 30)))
    return rgba, sh, hs


def compose(props, size, hotspot):
    """Flatten props [(dx, dy, rgba, shadow, (hx, hy))] into one sprite, back to front."""
    w, h = size
    out = np.zeros((h, w, 4), np.uint8)
    sh_out = np.zeros((h, w), np.uint8)
    for dx, dy, rgba, sh, (phx, phy) in sorted(props, key=lambda p: p[1]):
        ph, pw = rgba.shape[:2]
        dy = 0 if dy <= -1000 else dy   # -1000 marks a full-size background layer
        ox, oy = int(round(hotspot[0] + dx - phx)), int(round(hotspot[1] + dy - phy))
        x0, y0 = max(0, ox), max(0, oy)
        x1, y1 = min(w, ox + pw), min(h, oy + ph)
        if x0 >= x1 or y0 >= y1:
            continue
        src = rgba[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        m = src[..., 3] > 127
        out[y0:y1, x0:x1][m] = src[m]
        s = sh[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        np.maximum(sh_out[y0:y1, x0:x1], s, out=sh_out[y0:y1, x0:x1])
    sh_out[out[..., 3] > 0] = 0
    return out, sh_out, hotspot


def pumpkin_patch(left=1.0, variant=0, size=(112, 84)):
    """A forage bush as a pumpkin patch: vines and leaves with pumpkins that get picked
    as the patch is used up (left = 1, 0.66, 0.33). Leaves wilt as it empties."""
    rng = np.random.default_rng(700 + variant)
    hs = (size[0] // 2, size[1] - 26)
    ground = []
    for _ in range(5):   # dense spread of vines and leaves, drawn underneath everything
        a, sh, phs = leaves(rng, w=int(rng.uniform(44, 60)), h=int(rng.uniform(24, 32)), density=2.0)
        if left < 0.9:   # wilting: some leaves go yellow-brown
            wilt = (a[..., 3] > 0) & (rng.random(a.shape[:2]) < 0.6 * (1 - left))
            a[wilt, :3] = (a[wilt, :3] * 0.5 + np.array([150, 120, 56]) * 0.5).astype(np.uint8)
        ground.append((rng.uniform(-24, 24), rng.uniform(-10, 8), a, sh, phs))
    base = compose(ground, size, hs)
    total = 10
    kinds = ["plain"] * 4 + ["white", "jack", "gourd", "gourd", "gourd", "plain"]
    rng.shuffle(kinds)
    spots = [(rng.uniform(-38, 38), rng.uniform(-13, 13)) for _ in range(total)]
    n = max(1, int(round(total * left)))
    props = [(0, -1000, base[0], base[1], hs)]   # dy -1000 keeps the leaf layer at the back
    for (dx, dy), kind in list(zip(spots, kinds))[:n]:
        a, sh, phs = pumpkin(rng, kind)
        props.append((dx, dy, a, sh, phs))
    return compose(props, size, hs)
