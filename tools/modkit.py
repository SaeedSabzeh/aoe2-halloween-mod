"""Shared helpers for building AoE2 DE graphics mods from rendered sprites."""
import json
import os
import numpy as np
import sld

# Where the game is installed (set AOE2_DIR to override) and where local mods go
# (set AOE2_MODS to override; by default the first profile under ~/Games/Age of Empires 2 DE).
GAME_DIR = os.environ.get("AOE2_DIR", "D:/SteamLibrary/steamapps/common/AoE2DE").rstrip("/\\") + "/"
GAME_GFX = GAME_DIR + "resources/_common/drs/graphics/"


def _find_mods_local():
    base = os.path.join(os.path.expanduser("~"), "Games", "Age of Empires 2 DE")
    profiles = sorted(d for d in os.listdir(base) if d.isdigit() and d != "0") if os.path.isdir(base) else []
    return os.path.join(base, profiles[0] if profiles else "0", "mods", "local")


MODS_LOCAL = os.environ.get("AOE2_MODS") or _find_mods_local()
CANVAS = 200
HOT = 100


def original_frame_count(name):
    hdr, frames = sld.read(GAME_GFX + name + ".sld")
    return hdr, len(frames)


def _place(img, hotspot):
    """Pad img so it sits on the 200x200 canvas with hotspot at (100, 100) on a 4px grid."""
    hx, hy = hotspot
    x1, y1 = HOT - hx, HOT - hy
    px, py = x1 % 4, y1 % 4
    x1 -= px
    y1 -= py
    pad = [(py, 0), (px, 0)] + [(0, 0)] * (img.ndim - 2)
    img = np.pad(img, pad)
    h, w = img.shape[:2]
    pad = [(0, (-h) % 4), (0, (-w) % 4)] + [(0, 0)] * (img.ndim - 2)
    img = np.pad(img, pad)
    h, w = img.shape[:2]
    return img, (x1, y1, x1 + w, y1 + h)


def make_frame(rgba, hotspot, shadow=None, playercolor=None):
    main, rect = _place(rgba, hotspot)
    layers = {"main": dict(rect=rect, pixels=main)}
    if shadow is not None and shadow.any():
        sh, srect = _place(shadow, hotspot)
        layers["shadow"] = dict(rect=srect, pixels=sh)
    if playercolor is not None:
        pc, _ = _place(playercolor, hotspot)
        layers["playercolor"] = dict(pixels=pc)
    return dict(width=CANVAS, height=CANVAS, hotspot=(HOT, HOT), unk=0, layers=layers)


def write_graphic(mod_dir, name, frames_per_dir, render_frame, dirs=16):
    """Write <name>.sld matching the original's total frame count.

    render_frame(direction, i, n) -> frame dict, where i is the frame within that direction.
    Any frames past dirs * frames_per_dir (the originals have one extra) repeat frame 0.
    """
    hdr, total = original_frame_count(name)
    frames = []
    cache = {}
    for k in range(total):
        d, i = divmod(k, frames_per_dir)
        if d >= dirs:
            d, i = 0, 0
        key = (d, i)
        if key not in cache:
            cache[key] = render_frame(d, i, frames_per_dir)
        frames.append(cache[key])
    out = os.path.join(mod_dir, "resources", "_common", "drs", "graphics")
    os.makedirs(out, exist_ok=True)
    sld.write(os.path.join(out, name + ".sld"), hdr, frames)
    return total


def write_info(mod_dir, title, description):
    os.makedirs(mod_dir, exist_ok=True)
    with open(os.path.join(mod_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump({"Title": title, "Description": description, "Author": "SAEED"}, f, indent=2)
