"""Ketchup bottles (with a sauce puddle) to replace forage bushes."""
import numpy as np
import render3d
from render3d import Scene

BOTTLE = (206, 28, 22)
CAP = (245, 245, 240)
SAUCE = np.array([178, 20, 16])
SAUCE_HI = np.array([224, 70, 56])


def _bottle(s, x, y, tipped=0.0, h=30.0, r=6.0):
    """Upright squeeze bottle at (x, y); tipped=1 lays it on its side pointing +x."""
    def P(z_along):
        # point z_along up the bottle's axis, rotated down towards +x when tipped
        a = tipped * np.pi / 2
        return (x + np.sin(a) * z_along, y, r * np.sin(a) + np.cos(a) * z_along)
    s.capsule(P(r), P(h * 0.72), r, BOTTLE)  # start one radius up so it rests on the ground
    s.capsule(P(h * 0.78), P(h * 0.86), r * 0.6, CAP, r1=r * 0.5)  # cap
    s.capsule(P(h * 0.86), P(h * 1.0), r * 0.32, CAP, r1=r * 0.15)  # nozzle


def render(sauce_left=1.0, variant=0, size=(112, 84)):
    """Return (rgba, shadow, hotspot). Fewer bottles stand as the patch is used up."""
    rng = np.random.default_rng(500 + variant)
    s = Scene()
    n = 5
    # spread out on a jittered ring so bottles don't merge into one blob
    base_ang = rng.uniform(0, 2 * np.pi)
    spots = [(np.cos(base_ang + k * 2 * np.pi / n) * rng.uniform(13, 20),
              np.sin(base_ang + k * 2 * np.pi / n) * rng.uniform(13, 20)) for k in range(n)]
    standing = int(round(n * sauce_left))
    for k, (x, y) in enumerate(spots):
        if k < standing:
            _bottle(s, x, y, h=rng.uniform(27, 33))
        elif k < standing + 2:
            _bottle(s, x, y, tipped=1.0, h=28)  # emptied bottles lying in the sauce
    rgba, _, shadow, hs = render3d.render(s, 0.6 + variant, size, ground=26)

    # ketchup puddle on the ground under the bottles, bigger as the bottles empty
    h, w = rgba.shape[:2]
    ys, xs = np.mgrid[:h, :w].astype(float)
    hx, hy = hs
    puddle = np.zeros((h, w), bool)
    for _ in range(7):
        cx, cy = hx + rng.uniform(-28, 28), hy + rng.uniform(-10, 10)
        rx = rng.uniform(8, 14) * (1.3 - 0.5 * sauce_left)
        puddle |= ((xs - cx) / rx) ** 2 + ((ys - cy) / (rx * 0.5)) ** 2 <= 1
    under = puddle & (rgba[..., 3] == 0)
    rgba[under, :3] = SAUCE
    # glossy rim: lighter where the puddle edge faces the light (top-left)
    edge = under & ~np.roll(puddle, 2, 0)
    rgba[edge, :3] = SAUCE_HI
    rgba[under, 3] = 255
    shadow[under] = 0
    return rgba, shadow, hs
