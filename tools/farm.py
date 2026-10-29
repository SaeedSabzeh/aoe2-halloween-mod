"""Turn farm terrain textures (DXT1 .dds) into pumpkin fields."""
import numpy as np
from PIL import Image, ImageDraw
import sld

S = 4  # supersampling for drawing pumpkins


def read_dds(path):
    im = Image.open(path)
    im.load()
    return open(path, "rb").read(128), np.asarray(im.convert("RGBA")).copy()


def write_dds(path, header, rgba):
    """DXT1 with a full mip chain down to 1x1 (as the game's files have), reusing the original header."""
    out = [header]
    img = Image.fromarray(rgba)
    while True:
        a = np.asarray(img.convert("RGBA"))
        h, w = a.shape[:2]
        if w < 4 or h < 4:   # 2x2 and 1x1 levels still take one 4x4 block each
            a = np.pad(a, ((0, 4 - h), (0, 4 - w), (0, 0)), mode="edge")
        hb, wb = a.shape[0] // 4, a.shape[1] // 4
        blocks = a.reshape(hb, 4, wb, 4, 4).swapaxes(1, 2).reshape(-1, 4, 4, 4)
        out.append(sld.bc1_encode_many(blocks))
        if w <= 1 and h <= 1:
            break
        img = img.resize((max(1, w // 2), max(1, h // 2)), Image.BOX)
    open(path, "wb").write(b"".join(out))


def _row_centres(rgba):
    """Rows of crop run horizontally; find them from how green/bright each line is."""
    g = rgba[..., 1].astype(float) - rgba[..., 2].astype(float) * 0.6
    prof = g.mean(axis=1)
    prof = np.convolve(prof - prof.mean(), np.ones(9) / 9, mode="same")
    peaks = [y for y in range(len(prof)) if prof[y] == prof[max(0, y - 6):y + 7].max() and prof[y] > 0]
    return peaks


def vines(rgba, strength=1.0):
    """Recolour golden crop into dark pumpkin-vine greens, keeping the soil."""
    a = rgba[..., :3].astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    crop = np.clip((r + g) / 2 - b - 40, 0, 80) / 80 * strength    # yellow-ish = crop
    lum = (r + g + b) / 3
    target = np.stack([lum * 0.42, lum * 0.62, lum * 0.3], -1)
    a = a * (1 - crop[..., None]) + target * crop[..., None]
    rgba[..., :3] = np.clip(a, 0, 255).astype(np.uint8)
    return rgba


def pumpkins(rgba, seed, size=(34, 48), spacing=(90, 150), density=0.8, rotten=False, young=0.0):
    """Paint top-down pumpkins along the crop rows, wrapping at the edges so it tiles."""
    rng = np.random.default_rng(seed)
    H, W = rgba.shape[:2]
    canvas = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    rows = _row_centres(rgba)
    if len(rows) < 8:   # bare soil: no crop rows to find, use an even spacing
        rows = list(range(20, H, 43))
    if rotten:  # no crop rows left on a spent farm: scatter instead
        rows = list(range(0, H, int(spacing[0] * 1.3)))
    for y in rows:
        x = rng.uniform(0, spacing[1])
        while x < W:
            if rng.random() < density:
                r = rng.uniform(*size) / 2
                cx, cy = x + rng.uniform(-6, 6), y + rng.uniform(-6, 6)
                base = np.array([(206, 104, 26), (190, 88, 22), (220, 124, 38), (176, 80, 24)][rng.integers(4)])
                if rng.random() < 0.12:
                    base = np.array([222, 214, 188])   # the odd white pumpkin
                if rotten:
                    base = base * np.array([0.62, 0.5, 0.42])   # leftover, going bad
                if young:   # still growing: greener the younger it is
                    base = base * (1 - young) + np.array([92, 128, 52]) * young
                for ox in (-W, 0, W):          # wrap horizontally
                    for oy in (-H, 0, H):      # and vertically
                        _pumpkin_top(d, (cx + ox) * S, (cy + oy) * S, r * S, base, rng)
            x += rng.uniform(*spacing)
    layer = np.asarray(canvas.resize((W, H), Image.BOX)).astype(float)
    a = layer[..., 3:4] / 255
    out = rgba.astype(float)
    # soft contact shadow under each pumpkin, towards the bottom-right
    sh = np.roll(np.roll(a[..., 0], 6, 0), 5, 1)
    out[..., :3] *= (1 - 0.35 * np.clip(sh - a[..., 0], 0, 1))[..., None]
    out[..., :3] = out[..., :3] * (1 - a) + layer[..., :3] * a
    rgba[..., :3] = np.clip(out[..., :3], 0, 255).astype(np.uint8)
    return rgba


def _pumpkin_top(d, cx, cy, r, base, rng):
    """A pumpkin seen from above: ribbed, lit from the upper left, with a stem."""
    if cx + r < -r or cy + r < -r or cx - r > d.im.size[0] + r or cy - r > d.im.size[1] + r:
        return
    steps = 8
    for k in range(steps, 0, -1):     # concentric shading, brighter towards the light
        f = k / steps
        off = (1 - f) * r * 0.35
        c = np.clip(base * (0.62 + 0.5 * (1 - f)), 0, 255).astype(int)
        d.ellipse([cx - r * f - off, cy - r * f * 0.95 - off, cx + r * f - off, cy + r * f * 0.95 - off],
                  fill=tuple(c) + (255,))
    ribs = int(rng.integers(7, 10))
    for k in range(ribs):             # rib grooves
        a = k * 2 * np.pi / ribs + rng.uniform(-0.1, 0.1)
        d.line([(cx + np.cos(a) * r * 0.18, cy + np.sin(a) * r * 0.18),
                (cx + np.cos(a) * r * 0.95, cy + np.sin(a) * r * 0.9)],
               fill=tuple(np.clip(base * 0.55, 0, 255).astype(int)) + (255,), width=max(1, int(r * 0.07)))
    sr = r * 0.16
    d.ellipse([cx - sr, cy - sr, cx + sr, cy + sr], fill=(92, 80, 40, 255))
    d.line([(cx, cy), (cx + rng.uniform(-1, 1) * r * 0.35, cy - r * 0.3)], fill=(84, 74, 36, 255),
           width=max(1, int(r * 0.12)))
