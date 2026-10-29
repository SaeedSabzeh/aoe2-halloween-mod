"""Build the Halloween mod: spider sheep, skeleton boars, webbed trees."""
import math
import os
import sys
import re
import zlib
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
import modkit
import render3d
import creatures
import sld
import webs

MOD = os.path.join(modkit.MODS_LOCAL, "Halloween")
SPAN_TREES = ("n_tree_jungle_x1", "n_tree_rainforest_x1")
SKIP_TREES = ("stump", "felled", "reeds")  # left as the base game has them
PINE_VARIANTS = (3, 12)  # the dark green cone pines in n_tree_pine_x1
PINE_SCALE = 0.5         # small trees, so units and resources stay easy to see
TILE_HALF_W, TILE_HALF_H = 47, 24   # one map tile on screen, for the grid shadow
GRID_SHADOW = 26
TREE_HOT = 200
SKELETON_PIGS = ("a_hunt_boar_", "a_hunt_javelina_")  # javelina graphics are also used for bushpigs
GFX_OUT = os.path.join(MOD, "resources", "_common", "drs", "graphics")


def facing(d):
    return -d * math.pi / 8


def creature(builder, size, **kw):
    def render(d, i, n, pose):
        t = i / n
        extra = {}
        if pose == "dead":
            # death: collapse over the animation; decay: picked clean over the animation
            extra = dict(collapse=min(1.0, i / max(1, n - 1) * 1.3)) if kw.get("dying") else \
                dict(collapse=1.0, amount=1 - i / max(1, n - 1))
        scene = builder(phase=t, pose=pose, **extra)
        rgba, pc, sh, hs = render3d.render(scene, facing(d), size)
        return modkit.make_frame(rgba, hs, shadow=sh, playercolor=pc if pc.any() else None)
    return render


def anim(builder, size, pose, dying=False):
    r = creature(builder, size, dying=dying)
    return lambda d, i, n: r(d, i, n, pose)


def _place(img, hotspot, hot=TREE_HOT):
    """Pad img onto a (2*hot)^2 canvas with hotspot at (hot, hot), on the 4px grid."""
    hx, hy = hotspot
    x1, y1 = hot - hx, hot - hy
    px, py = x1 % 4, y1 % 4
    pad = [(py, 0), (px, 0)] + [(0, 0)] * (img.ndim - 2)
    img = np.pad(img, pad)
    h, w = img.shape[:2]
    img = np.pad(img, [(0, (-h) % 4), (0, (-w) % 4)] + [(0, 0)] * (img.ndim - 2))
    h, w = img.shape[:2]
    return img, (x1 - px, y1 - py, x1 - px + w, y1 - py + h)


def _scale_layer(pixels, rect, hotspot, f, rgba=True):
    """Scale a layer about the frame hotspot. Returns (pixels, hotspot within pixels)."""
    x1, y1, _, _ = rect
    hx, hy = hotspot[0] - x1, hotspot[1] - y1
    h, w = pixels.shape[:2]
    nw, nh = max(4, int(round(w * f))), max(4, int(round(h * f)))
    if rgba:
        a = pixels[..., 3:4].astype(np.float64) / 255
        pre = Image.fromarray(np.clip(pixels[..., :3] * a, 0, 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
        al = np.asarray(Image.fromarray(pixels[..., 3]).resize((nw, nh), Image.BOX), dtype=np.float64) / 255
        rgb = np.asarray(pre, dtype=np.float64) / np.maximum(al[..., None], 1e-3)
        out = np.zeros((nh, nw, 4), np.uint8)
        out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        out[..., 3] = np.where(al >= 0.5, 255, 0)
        out[out[..., 3] == 0, :3] = 0
    else:
        out = np.asarray(Image.fromarray(pixels).resize((nw, nh), Image.BOX))
    return out, (hx * f, hy * f)


def _grid_shadow(tree_shadow, tree_hot, value=None):
    """A tile-sized diamond under the tree, merged with the tree's own shadow."""
    value = GRID_SHADOW if value is None else value
    th, tw = tree_shadow.shape
    left = int(max(TILE_HALF_W + 1, np.ceil(tree_hot[0])))
    top = int(max(TILE_HALF_H + 1, np.ceil(tree_hot[1])))
    W = left + int(max(TILE_HALF_W + 1, tw - tree_hot[0])) + 1
    H = top + int(max(TILE_HALF_H + 1, th - tree_hot[1])) + 1
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon([(left, top - TILE_HALF_H), (left + TILE_HALF_W, top),
                                 (left, top + TILE_HALF_H), (left - TILE_HALF_W, top)], fill=value)
    out = np.asarray(img).copy()
    ox, oy = int(round(left - tree_hot[0])), int(round(top - tree_hot[1]))
    region = out[oy:oy + th, ox:ox + tw]
    np.maximum(region, tree_shadow[:region.shape[0], :region.shape[1]], out=region)
    return out, (left, top)


def web_trees():
    """Every tree becomes a small webbed pine with a tile grid shadow, built from the base
    game's own pine art (scaled down), so the mod contains no other mod's work."""
    _, pines = sld.read(modkit.GAME_GFX + "n_tree_pine_x1.sld")
    small = []
    for fr in pines[PINE_VARIANTS[0]:PINE_VARIANTS[1]]:
        m, s = fr["layers"]["main"], fr["layers"]["shadow"]
        px, phs = _scale_layer(m["pixels"], m["rect"], fr["hotspot"], PINE_SCALE)
        sp, shs = _scale_layer(s["pixels"], s["rect"], fr["hotspot"], PINE_SCALE, rgba=False)
        sp, shs = _grid_shadow(sp, shs)
        small.append((px, phs, sp, shs))
    os.makedirs(GFX_OUT, exist_ok=True)
    for f in sorted(os.listdir(modkit.GAME_GFX)):
        if not (f.startswith("n_tree_") and f.endswith("_x1.sld")) or any(k in f for k in SKIP_TREES):
            continue
        name = f[:-4]
        _, total = modkit.original_frame_count(name)
        out = []
        for k in range(total):
            px, phs, sp, shs = small[k % len(small)]
            pp, rect = _place(px, (int(round(phs[0])), int(round(phs[1]))))
            L = webs.web_tree(webs.Layer(pp, rect), (TREE_HOT, TREE_HOT),
                              seed=zlib.crc32(name.encode()) % 10000 + k, span=name in SPAN_TREES)
            spp, srect = _place(sp, shs)
            out.append(dict(width=2 * TREE_HOT, height=2 * TREE_HOT, hotspot=(TREE_HOT, TREE_HOT), unk=0,
                            layers={"main": dict(rect=L.rect(), pixels=L.px),
                                    "shadow": dict(rect=srect, pixels=spp)}))
        sld.write(os.path.join(GFX_OUT, f), {}, out)
        print(f"{name}: {len(out)} frames", flush=True)


def town_centers():
    """Jack-o'-lanterns along the front of every Town Center (whole sprite and front piece)."""
    import re
    import tc_pumpkins
    os.makedirs(GFX_OUT, exist_ok=True)
    pat = re.compile(r"^b_[a-z]+_town_center_age\d(_front)?_x1\.sld$")
    for k, f in enumerate(sorted(os.listdir(modkit.GAME_GFX))):
        if not pat.match(f):
            continue
        hdr, frames = sld.read(modkit.GAME_GFX + f)
        for fr in frames:
            tc_pumpkins.decorate(fr, variant=k)
        sld.write(os.path.join(GFX_OUT, f), hdr, frames)
        print(f"{f[:-4]}: {len(frames)} frames", flush=True)


def herd_spiders():
    """Every herd animal (not just sheep) becomes a spider sheep, since each map and
    civilization starts with its own herdable."""
    S = creatures.spider_sheep
    for f in sorted(os.listdir(modkit.GAME_GFX)):
        if not (f.startswith("a_herd_") and f.endswith("_x1.sld")):
            continue
        name = f[:-4]
        if "decay" in name:
            fn = anim(S, (52, 44), "dead")
        elif "death" in name:
            fn = anim(S, (52, 44), "dead", dying=True)
        elif "walk" in name or "run" in name:
            fn = anim(S, (52, 44), "walk")
        else:
            fn = anim(S, (52, 44), "idle")
        _, total = modkit.original_frame_count(name)
        print(f"{name}: {modkit.write_graphic(MOD, name, total // 16, fn)} frames", flush=True)


def ghost_deer():
    """Every huntable except boars/javelinas (skeletons) becomes a ghost of itself: the
    game's own animation, recoloured and see-through. Maps swap deer for hares, ostriches,
    zebras etc., so all of them get it."""
    import re
    import ghost
    animals = {}
    for f in sorted(os.listdir(modkit.GAME_GFX)):
        if f.startswith("a_hunt_") and f.endswith("_x1.sld") and not f.startswith(SKELETON_PIGS):
            key = re.sub(r"_(idle|walk|run|death|decay|attack)[A-Za-z]*_x1\.sld$", "", f)
            animals.setdefault(key, []).append(f)
    for key, files in animals.items():
        try:
            loaded = [(f,) + sld.read(modkit.GAME_GFX + f) for f in files]
        except Exception as e:  # an unusual file layout: leave the whole animal untouched
            print(f"{key}: skipped ({type(e).__name__})", flush=True)
            continue
        for f, hdr, frames in loaded:
            per = max(1, len(frames) // 16)
            out = []
            for k, fr in enumerate(frames):
                t = (k % per) / per
                if "death" in f:
                    opacity = 1 - t * 0.25         # dims a little as it falls
                elif "decay" in f:
                    opacity = 0.85 - 0.4 * t        # the carcass stays clearly visible, fading as it's eaten
                else:
                    opacity = 1.0
                out.append(ghost.ghostify(fr, opacity))
            sld.write(os.path.join(GFX_OUT, f), hdr, out)
            print(f"{f[:-4]}: {len(out)} frames", flush=True)


def skeleton_pigs():
    """Boars and javelinas (bushpigs share the javelina graphics) become skeleton boars."""
    B = creatures.skeleton_boar
    for f in sorted(os.listdir(modkit.GAME_GFX)):
        if not (f.startswith(SKELETON_PIGS) and f.endswith("_x1.sld")):
            continue
        name = f[:-4]
        if "decay" in name:
            fn = anim(B, (80, 64), "dead")
        elif "death" in name:
            fn = anim(B, (80, 64), "dead", dying=True)
        elif "attack" in name:
            fn = anim(B, (80, 64), "attack")
        elif "walk" in name or "run" in name:
            fn = anim(B, (80, 64), "walk")
        else:
            fn = anim(B, (80, 64), "idle")
        _, total = modkit.original_frame_count(name)
        print(f"{name}: {modkit.write_graphic(MOD, name, total // 16, fn)} frames", flush=True)


def pumpkin_patches():
    """Every forage bush (berry, fruit, papaya, pineapple) becomes a pumpkin patch."""
    import harvest
    for f in sorted(os.listdir(modkit.GAME_GFX)):
        m = re.match(r"^n_forage_[a-z]+(_(33|66))?_x1\.sld$", f)
        if not m:
            continue
        left = int(m.group(2)) / 100 if m.group(2) else 1.0
        name = f[:-4]
        _, total = modkit.original_frame_count(name)

        def render(d, i, n, left=left, name=name):
            rgba, sh, _ = harvest.pumpkin_patch(left, variant=i + zlib.crc32(name.encode()) % 50)
            # the game's bushes sit with their visual centre ~11px above the tile centre
            ys, xs = np.nonzero(rgba[..., 3] > 127)
            return modkit.make_frame(rgba, (int(xs.mean()), int(ys.mean()) + 11), shadow=sh)
        print(f"{name}: {modkit.write_graphic(MOD, name, total, render, dirs=1)} frames", flush=True)


def _haunt_trap(fr, seed):
    """Cobwebs across a fish trap's net, kept inside its box."""
    m = fr["layers"]["main"]
    L = webs.Layer(m["pixels"], m["rect"])
    L.grow = lambda *a: None          # never resize: masks and outline data stay valid
    xs, ys = webs.opaque_points(L)
    if len(xs) == 0:
        return fr
    l, r, t, b = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = r - l, b - t
    rng = np.random.default_rng(seed)  # same seed every frame: the web moves with the trap
    for poly in ([(l + .15 * w, t + .25 * h), (l + .6 * w, t + .05 * h), (l + .5 * w, b - .15 * h)],
                 [(l + .45 * w, t + .2 * h), (r - .05 * w, t + .3 * h), (l + .7 * w, b - .05 * h)]):
        webs.gauze(L, poly, rng, strands=12, strength=0.32, sag=2.0)
    m["pixels"] = L.px
    return fr


def haunted_water():
    """Ghost fish (all types, above and under water) and cobwebbed fish traps."""
    import ghost
    for f in sorted(os.listdir(modkit.GAME_GFX)):
        if not f.endswith("_x1.sld"):
            continue
        if f.startswith("a_fish_"):
            hdr, frames = sld.read(modkit.GAME_GFX + f)
            out = [ghost.ghostify(fr, opacity=0.85, float_px=0, wisp=False, shadow=0) for fr in frames]
        elif f.startswith("b_misc_fish_trap") and "foundation" not in f:
            hdr, frames = sld.read(modkit.GAME_GFX + f)
            out = [_haunt_trap(fr, 77) for fr in frames]
        else:
            continue
        sld.write(os.path.join(GFX_OUT, f), hdr, out)
        print(f"{f[:-4]}: {len(out)} frames", flush=True)


def _paste(fr, rgba, ox, oy):
    """Paste a prop into a frame's main layer (top-left inside the box); clears player colour."""
    m = fr["layers"]["main"]
    px = m["pixels"]
    h, w = rgba.shape[:2]
    if ox < 0 or oy < 0 or ox + w > px.shape[1] or oy + h > px.shape[0]:
        return False
    sel = rgba[..., 3] > 127
    px[oy:oy + h, ox:ox + w][sel] = rgba[sel]
    pc = fr["layers"].get("playercolor")
    if pc is not None:
        pc["pixels"][oy:oy + h, ox:ox + w][sel] = 0
    return True


def _skull(rng):
    s = render3d.Scene()
    s.sphere((0, 0, 2.4), 2.3, (214, 202, 172))
    s.sphere((1.2, 0, 1.3), 1.2, (176, 162, 130))
    for side in (1, -1):
        s.sphere((1.9, side * 0.85, 2.6), 0.75, (46, 34, 28))
    rgba, _, _, hs = render3d.render(s, -np.pi / 2 + rng.uniform(-0.5, 0.5), (12, 12), ground=2,
                                     outline=None, soft=True, noise=0.04, seed=int(rng.integers(1 << 30)))
    ys, xs = np.nonzero(rgba[..., 3] > 127)
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pastures():
    """Yurt with cobwebs and lanterns, skulls on the fence posts, spider pasture ibex."""
    import harvest
    rng = np.random.default_rng(31)
    hdr, frames = sld.read(modkit.GAME_GFX + "b_dark_pasture_x1.sld")
    for fr in frames:
        m = fr["layers"]["main"]
        L = webs.Layer(m["pixels"], m["rect"])
        L.grow = lambda *a: None
        xs, ys = webs.opaque_points(L)
        l, r, t, b = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = r - l, b - t
        # draped over the front of the dome and down one side
        for poly in ([(l + .38 * w, t + .04 * h), (l + .16 * w, t + .5 * h), (l + .48 * w, t + .58 * h)],
                     [(l + .56 * w, t + .06 * h), (l + .42 * w, t + .46 * h), (l + .78 * w, t + .42 * h)],
                     [(l + .2 * w, t + .3 * h), (l + .1 * w, t + .7 * h), (l + .32 * w, t + .72 * h)]):
            webs.gauze(L, poly, rng, strands=24, strength=0.35, sag=2.5)
        m["pixels"] = L.px
        x1, y1, _, _ = m["rect"]
        for fx, size_kind in ((0.42, "jack"), (0.62, "jack")):
            rgba, _, (phx, phy) = harvest.pumpkin(rng, size_kind, R=5.0)
            ys_, xs_ = np.nonzero(rgba[..., 3] > 127)
            rgba = rgba[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
            _paste(fr, rgba, int(l + fx * w - x1 - rgba.shape[1] / 2), int(b - 4 - y1 - rgba.shape[0]))
    sld.write(os.path.join(GFX_OUT, "b_dark_pasture_x1.sld"), hdr, frames)
    print("b_dark_pasture_x1: yurt", flush=True)

    hdr, frames = sld.read(modkit.GAME_GFX + "b_dark_pasture_corner_posts_x1.sld")
    for fr in frames:
        m = fr["layers"]["main"]
        ys, xs = np.nonzero(m["pixels"][..., 3] > 127)
        if len(ys) == 0:
            continue
        top = ys.min()
        cx = int(xs[ys <= top + 2].mean())
        sk = _skull(rng)
        _paste(fr, sk, cx - sk.shape[1] // 2, max(0, top - 1))
    sld.write(os.path.join(GFX_OUT, "b_dark_pasture_corner_posts_x1.sld"), hdr, frames)
    print("b_dark_pasture_corner_posts_x1: skulls", flush=True)

    S = creatures.spider_sheep
    for name, fn in (("a_pasture_ibex_idleA_x1", anim(S, (52, 44), "idle")),
                     ("a_pasture_ibex_deathA_x1", anim(S, (52, 44), "dead", dying=True)),
                     ("a_pasture_ibex_decayA_x1", anim(S, (52, 44), "dead"))):
        _, total = modkit.original_frame_count(name)
        print(f"{name}: {modkit.write_graphic(MOD, name, total // 16, fn)} frames", flush=True)


def pumpkin_farms():
    """Farm crop texture becomes a pumpkin field; spent farms keep a few rotting pumpkins."""
    import farm
    src = os.path.join(os.path.dirname(os.path.dirname(modkit.GAME_GFX.rstrip("/"))), "terrain", "textures", "2x")
    dst = os.path.join(MOD, "resources", "_common", "terrain", "textures", "2x")
    os.makedirs(dst, exist_ok=True)
    # farms grow through fc1 (plowed) -> fc2 -> fc3 -> fm1 (grown), and fm2 when spent
    stages = (("g_fc2.dds", dict(seed=1, radius=(2.8, 3.6), young=0.85)),
              ("g_fc3.dds", dict(seed=1, radius=(4.2, 5.4), young=0.35)),
              ("g_fm1.dds", dict(seed=1)),
              ("g_fm2.dds", dict(seed=2, density=0.35, radius=(4.5, 6.0), rotten=True)))
    for name, kw in stages:
        header, rgba = farm.read_dds(os.path.join(src, name))
        rgba = farm.pumpkins3d(rgba, **kw)
        farm.write_dds(os.path.join(dst, name), header, rgba)
        print(f"{name}: pumpkin field", flush=True)


DESCRIPTION = ("A full Halloween makeover for every map. Herd animals become spider sheep, wild game "
               "becomes ghosts of itself, boars are skeletons, forests are small pines draped in cobwebs "
               "(strung between trees in jungles), forage bushes are pumpkin patches, farms grow pumpkins, "
               "fish are ghosts, fish traps and pastures are haunted, and every Town Center gets a harvest "
               "display with jack-o'-lanterns and a skeleton. Graphics only: works in multiplayer and "
               "ranked. Don't combine with other mods that replace the same animals or trees.")


if __name__ == "__main__":
    modkit.write_info(MOD, "Halloween", DESCRIPTION)
    herd_spiders()
    ghost_deer()
    skeleton_pigs()
    pumpkin_patches()
    haunted_water()
    pastures()
    pumpkin_farms()
    web_trees()
    town_centers()
    print("done ->", MOD)
