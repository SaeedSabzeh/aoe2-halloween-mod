"""Walking pizza boxes that pop open into a pizza when killed (poly3d scenes)."""
import numpy as np
from poly3d import Scene, rot_y, rot_z

TAU = 2 * np.pi

KRAFT = (204, 168, 118)
KRAFT_DARK = (170, 134, 90)
KRAFT_INSIDE = (226, 204, 162)
WHITE_BOX = (238, 234, 226)
WHITE_BOX_SIDE = (214, 208, 198)
LOGO_RED = (200, 36, 30)
LOGO_WHITE = (250, 246, 238)
LOGO_GREEN = (52, 140, 60)

CRUST = (214, 150, 72)

# name: (sauce, cheese, [(topping colour, radius as fraction of pizza, count)])
PIZZAS = {
    "pepperoni":   ((196, 48, 32), (250, 212, 96), [((170, 34, 30), 0.15, 8)]),
    "margherita":  ((200, 50, 34), (250, 220, 120), [((252, 248, 236), 0.17, 5), ((50, 140, 48), 0.08, 6)]),
    "hawaiian":    ((198, 52, 36), (250, 214, 104), [((240, 200, 60), 0.09, 9), ((236, 150, 150), 0.11, 7)]),
    "bbq_chicken": ((112, 46, 26), (246, 206, 100), [((236, 214, 170), 0.1, 8), ((120, 40, 110), 0.06, 8)]),
    "mushroom":    ((196, 52, 36), (248, 216, 112), [((170, 150, 128), 0.11, 9)]),
    "meat_lovers": ((190, 44, 30), (246, 204, 92), [((170, 34, 30), 0.13, 9), ((120, 70, 44), 0.08, 10), ((236, 150, 150), 0.09, 6)]),
    "bbq":         ((104, 40, 22), (240, 196, 90), [((150, 84, 44), 0.11, 10), ((236, 230, 220), 0.07, 8)]),
    "veggie":      ((196, 52, 36), (250, 214, 104), [((60, 150, 56), 0.09, 8), ((230, 60, 40), 0.08, 6), ((40, 36, 40), 0.06, 8)]),
}

# box size (width, height) and pizza per animal; big animals get big boxes
ANIMALS = {
    "sheep": (24, 5, "pepperoni", KRAFT), "goat": (24, 5, "margherita", KRAFT),
    "argali": (24, 5, "margherita", KRAFT), "pig": (26, 5, "hawaiian", KRAFT),
    "turkey": (24, 5, "bbq_chicken", KRAFT), "goose": (24, 5, "bbq_chicken", KRAFT),
    "chicken_black": (16, 4, "bbq_chicken", KRAFT), "chicken_brown": (16, 4, "bbq_chicken", KRAFT),
    "chicken_white": (16, 4, "bbq_chicken", KRAFT), "capybara": (24, 5, "mushroom", KRAFT),
    "cow_black": (36, 7, "meat_lovers", WHITE_BOX), "cow_blackandwhite": (36, 7, "meat_lovers", WHITE_BOX),
    "cow_brown": (36, 7, "meat_lovers", WHITE_BOX), "cow_brownandwhite": (36, 7, "meat_lovers", WHITE_BOX),
    "buffalo": (38, 7, "bbq", WHITE_BOX), "llama": (34, 7, "veggie", WHITE_BOX),
    "llama_white": (34, 7, "veggie", WHITE_BOX),
}


def pizza(s, r, kind, slices_left=8, z=0.0, rotation=0.0, bias=0.0):
    sauce, cheese, tops = PIZZAS[kind]
    if slices_left <= 0:
        return
    a1 = rotation + slices_left * TAU / 8
    s.disc((0, 0, z), r, CRUST, a0=rotation, a1=a1, bias=bias)
    s.disc((0, 0, z + 0.05), r * 0.86, sauce, a0=rotation, a1=a1, bias=bias + 0.01)
    s.disc((0, 0, z + 0.1), r * 0.8, cheese, a0=rotation, a1=a1, bias=bias + 0.02)
    rng = np.random.default_rng(hash(kind) % 1000)
    for color, frac, count in tops:
        for _ in range(count):
            ang = rng.uniform(0, TAU)
            rad = r * np.sqrt(rng.uniform(0.04, 0.5))
            if (ang % TAU) > slices_left * TAU / 8 - 0.15:
                continue
            a = ang + rotation
            s.disc((np.cos(a) * rad, np.sin(a) * rad, z + 0.15), r * frac, color, segments=8, bias=bias + 0.03)


def box(s, w, h, lid_angle, color, pizza_kind=None, slices_left=8, pizza_rot=0.0):
    """Pizza box centred on the origin. lid_angle 0 = closed, ~1.9 = wide open."""
    side = KRAFT_DARK if color == KRAFT else WHITE_BOX_SIDE
    x0, x1, y0, y1 = -w / 2, w / 2, -w / 2, w / 2
    # tray: inside floor and four walls (two-sided so the inside shows when open)
    s.poly([(x0, y0, 0.3), (x1, y0, 0.3), (x1, y1, 0.3), (x0, y1, 0.3)], KRAFT_INSIDE, two_sided=True, bias=-0.2)
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        s.poly([(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], h), (a[0], a[1], h)], side, two_sided=True)
    if pizza_kind is not None and lid_angle > 0.3:
        pizza(s, w * 0.42, pizza_kind, slices_left, z=0.6, rotation=pizza_rot, bias=0.3)
    # lid hinged on the back edge (x = x0), lifting its front edge up
    s.push(rot_y(-lid_angle), t=(x0, 0, h))
    lw = w + 0.6
    lid = [(0, y0 - 0.3, 0.2), (lw, y0 - 0.3, 0.2), (lw, y1 + 0.3, 0.2), (0, y1 + 0.3, 0.2)]
    s.poly(lid, color, two_sided=True)
    # logo on the lid: player-coloured ring, white middle, a red slice
    c = (lw / 2, 0, 0.25)
    s.disc(c, w * 0.3, LOGO_RED, pc=1.0, bias=0.05)
    s.disc((c[0], c[1], 0.3), w * 0.23, LOGO_WHITE, bias=0.06)
    s.disc((c[0], c[1], 0.35), w * 0.18, LOGO_RED, a0=-0.45, a1=0.45, bias=0.07)
    s.disc((c[0] + w * 0.05, c[1], 0.4), w * 0.04, LOGO_GREEN, segments=6, bias=0.08)
    # front lip of the lid, also player coloured, so the closed box reads as owned
    s.poly([(lw, y0 - 0.3, 0.2), (lw, y1 + 0.3, 0.2), (lw, y1 + 0.3, -1.4), (lw, y0 - 0.3, -1.4)],
           color, pc=1.0, two_sided=True)
    s.pop()


def animal(kind, pose, phase=0.0, progress=0.0):
    """Scene for a herd animal. pose: idle, walk, run, death, decay.

    progress 0..1 through death/decay animations; phase 0..1 through looping ones.
    """
    w, h, pz, color = ANIMALS[kind]
    s = Scene()
    lift, tilt, lid, slices, scale = 0.0, 0.0, 0.0, 8, 1.0
    if pose == "idle":
        # the lid peeks open now and then
        lid = 0.25 * max(0.0, np.sin(phase * TAU * 2)) ** 6
    elif pose in ("walk", "run"):
        hops = 2 if pose == "walk" else 3
        hop = abs(np.sin(phase * np.pi * hops))
        lift = hop * (w * (0.2 if pose == "walk" else 0.3))
        tilt = 0.18 * np.cos(phase * TAU * hops)
    elif pose == "death":
        t = progress
        lift = w * 0.4 * np.sin(np.pi * min(1, t / 0.45)) if t < 0.45 else 0.0
        lid = 0.0 if t < 0.45 else min(1.0, (t - 0.45) / 0.35) * 1.95
    elif pose == "decay":
        lid = 1.95
        slices = int(np.ceil(8 * max(0.0, 1 - progress / 0.85)))
        if progress > 0.85:
            scale = 1 - (progress - 0.85) / 0.15 * 0.55  # empty box crumples away
    s.push(rot_y(tilt), t=(0, 0, lift))
    s.push(np.eye(3) * scale)
    box(s, w, h, lid, color, pizza_kind=pz if pose in ("death", "decay") else None, slices_left=slices)
    s.pop()
    s.pop()
    return s
