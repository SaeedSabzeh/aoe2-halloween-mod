"""Build the Pizza Sheep mod: sheep -> pizzas, forage bushes -> ketchup bottles."""
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import modkit
import pizza
import ketchup

MOD = os.path.join(modkit.MODS_LOCAL, "Pizza Sheep")
RADIUS = 17


def facing(d):
    # direction 0 faces screen-right, directions advance clockwise
    return -d * math.pi / 8


def sheep_frame(rgba, pc, sh, hs):
    return modkit.make_frame(rgba, hs, shadow=sh, playercolor=pc)


def idle(d, i, n):
    t = i / n
    lift = 0.8 * (1 - math.cos(2 * math.pi * t)) / 2
    return sheep_frame(*pizza.render(RADIUS, rotation=facing(d), lift=lift))


def walk(d, i, n):
    t = i / n
    lift = 5.0 * abs(math.sin(2 * math.pi * t))  # two hops per cycle
    spin = facing(d) + 0.15 * math.sin(2 * math.pi * t)
    return sheep_frame(*pizza.render(RADIUS, rotation=spin, lift=lift))


def death(d, i, n):
    t = i / max(1, n - 1)
    lift = 10.0 * math.sin(math.pi * min(1.0, t / 0.6)) if t < 0.6 else 0.0
    spin = facing(d) + t * math.pi  # flips around as it falls
    return sheep_frame(*pizza.render(RADIUS, rotation=spin, lift=lift))


def decay(d, i, n):
    # carcass being eaten: slices disappear, last frames are a lone shrinking slice
    t = i / max(1, n - 1)
    slices = max(1, math.ceil(8 * (1 - t)))
    r = RADIUS if t < 0.85 else RADIUS * (1 - (t - 0.85) / 0.15 * 0.6)
    return sheep_frame(*pizza.render(r, rotation=facing(d) + math.pi, slices_left=slices))


def bush(left):
    def render(d, i, n):
        rgba, sh, _ = ketchup.render(sauce_left=left, variant=i)
        # the game's bushes sit with their visual centre ~11px above the tile centre
        ys, xs = np.nonzero(rgba[..., 3] > 127)
        return modkit.make_frame(rgba, (int(xs.mean()), int(ys.mean()) + 11), shadow=sh)
    return render


if __name__ == "__main__":
    modkit.write_info(MOD, "Pizza Sheep", "Sheep become pizzas and forage bushes become ketchup bottles.")
    jobs = [
        ("a_herd_sheep_idleA_x1", 60, idle),
        ("a_herd_sheep_walkA_x1", 30, walk),
        ("a_herd_sheep_deathA_x1", 30, death),
        ("a_herd_sheep_decayA_x1", 30, decay),
        ("n_forage_bush_x1", 4, bush(1.0)),
        ("n_forage_bush_66_x1", 4, bush(0.66)),
        ("n_forage_bush_33_x1", 4, bush(0.33)),
    ]
    for name, per, fn in jobs:
        dirs = 1 if name.startswith("n_") else 16
        n = modkit.write_graphic(MOD, name, per, fn, dirs=dirs)
        print(f"{name}: {n} frames", flush=True)
    print("done ->", MOD)
