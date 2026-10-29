"""Procedural pizza sprite, seen from the game's 2:1 isometric camera."""
import numpy as np

SS = 4  # supersampling factor

CRUST = np.array([214, 150, 72])
CRUST_DARK = np.array([168, 104, 44])
SAUCE = np.array([196, 48, 32])
CHEESE = np.array([250, 212, 96])
CHEESE_HI = np.array([255, 236, 150])
PEPPERONI = np.array([170, 34, 30])
PEPPERONI_HI = np.array([205, 70, 55])
BASIL = np.array([60, 140, 50])

# fixed topping layout in pizza space (unit disc): (r, angle, kind)
_rng = np.random.default_rng(7)
TOPPINGS = []
for k in range(9):
    TOPPINGS.append((0.25 + 0.5 * _rng.random(), _rng.random() * 2 * np.pi, "pep"))
for k in range(5):
    TOPPINGS.append((0.2 + 0.55 * _rng.random(), _rng.random() * 2 * np.pi, "basil"))


def render(radius, rotation=0.0, slices_left=8, lift=0.0, tilt=0.5, thickness=3.0):
    """Return (rgba HxWx4, playercolor_mask HxW, shadow HxW, hotspot (x, y)).

    radius     pizza radius in screen pixels (horizontal)
    rotation   spin of the pizza around its vertical axis, radians
    slices_left  0..8, missing slices are cut out of the pizza
    lift       pixels the pizza floats above its shadow (for hops/bobs)
    tilt       vertical squash of the disc (0.5 = flat on the ground in 2:1 iso)
    """
    pad = 3
    w = int(np.ceil(2 * radius)) + 2 * pad
    h = int(np.ceil(2 * radius * tilt + thickness + lift)) + 2 * pad
    w += (-w) % 4
    h += (-h) % 4
    W, H = w * SS, h * SS
    cx = W / 2
    cy_ground = H - (pad + radius * tilt) * SS          # centre of the shadow ellipse
    cy = cy_ground - (lift + thickness) * SS              # centre of the pizza's top face
    R = radius * SS
    ys, xs = np.mgrid[:H, :W].astype(np.float64)

    def disc_coords(cy_):
        dx = (xs - cx) / R
        dy = (ys - cy_) / (R * tilt)
        r = np.hypot(dx, dy)
        a = (np.arctan2(dy, dx) - rotation) % (2 * np.pi)
        return r, a, dx, dy

    def slice_mask(a):
        if slices_left >= 8:
            return np.ones_like(a, bool)
        return a < slices_left * (2 * np.pi / 8)

    rgb = np.zeros((H, W, 3))
    alpha = np.zeros((H, W))
    pcm = np.zeros((H, W))

    # crust side wall: stack of discs from bottom up to top face
    steps = max(2, int(thickness * SS))
    for i in range(steps + 1):
        cy_i = cy + (thickness * SS) * (1 - i / steps)
        r, a, _, _ = disc_coords(cy_i)
        m = (r <= 1.0) & slice_mask(a)
        rgb[m] = CRUST_DARK
        alpha[m] = 1
        pcm[m] = 0.55

    # top face
    r, a, dx, dy = disc_coords(cy)
    face = (r <= 1.0) & slice_mask(a)
    rim = face & (r > 0.82)
    inner = face & ~rim
    rgb[rim] = CRUST
    pcm[rim] = 1.0
    rgb[inner] = CHEESE
    pcm[inner] = 0
    # sauce peeking out near the rim
    rgb[inner & (r > 0.76)] = SAUCE
    # cheese highlights
    blob = np.sin(dx * 13 + 1.3) * np.sin(dy * 11 + 0.4) > 0.55
    rgb[inner & blob] = CHEESE_HI
    # slice cut lines
    seg = (a % (2 * np.pi / 8))
    cut = inner & ((seg < 0.035 / np.maximum(r, 0.05)) | (seg > 2 * np.pi / 8 - 0.035 / np.maximum(r, 0.05)))
    rgb[cut] = CHEESE * 0.8

    # toppings
    for tr, ta, kind in TOPPINGS:
        ang = ta + rotation
        tx = cx + np.cos(ang) * tr * R
        ty = cy + np.sin(ang) * tr * R * tilt
        if not slice_mask(np.array([ta % (2 * np.pi)]))[0]:
            continue
        size = 0.15 if kind == "pep" else 0.07
        d = np.hypot((xs - tx) / (size * R), (ys - ty) / (size * R * tilt))
        m = inner & (d <= 1)
        if kind == "pep":
            rgb[m] = PEPPERONI
            rgb[m & (d < 0.45) & (ys < ty)] = PEPPERONI_HI
        else:
            rgb[m] = BASIL

    # downsample
    def down(x):
        return x.reshape(h, SS, w, SS, *x.shape[2:]).mean(axis=(1, 3))
    a_d = down(alpha)
    rgb_d = down(rgb * alpha[..., None]) / np.maximum(a_d[..., None], 1e-6)
    pc_d = down(pcm * alpha) / np.maximum(a_d, 1e-6)

    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.clip(rgb_d, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(a_d >= 0.5, 255, 0)
    pc_mask = np.where(a_d >= 0.5, np.clip(pc_d * 255, 0, 255), 0).astype(np.uint8)

    # shadow: soft ellipse on the ground
    sdx = (xs - cx) / (R * 1.02)
    sdy = (ys - cy_ground) / (R * tilt * 1.02)
    sr = np.hypot(sdx, sdy)
    sa = (np.arctan2(sdy, sdx) - rotation) % (2 * np.pi)
    sh = down(np.clip(1.2 - sr, 0, 1) * (sr <= 1.0) * slice_mask(sa))
    shadow = (sh * 140).astype(np.uint8)
    shadow[out[..., 3] > 0] = 0

    hotspot = (w // 2, int(round(cy_ground / SS)))
    return out, pc_mask, shadow, hotspot
