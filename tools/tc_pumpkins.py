"""Autumn harvest displays (pumpkins, hay, candles, a skeleton) in front of Town Centers."""
import numpy as np
import harvest


def _layout(rng, flip):
    """Props as (dx, dy, lift, kind). dx/dy are ground spots relative to the building's
    hotspot; lift raises a prop onto something (a pumpkin on a hay bale)."""
    j = lambda v: v + rng.uniform(-3, 3)
    P = []
    # left corner: hay bale with pumpkins on it and around it, on a vine patch
    P += [(j(-108), j(22), 0, "leaves"), (j(-118), j(10), 0, "bale")]
    P += [(-118 + rng.uniform(-4, 4), 10.5, "BALE", "jack")]
    P += [(j(-96), j(24), 0, rng.choice(["jack", "plain"])), (j(-128), j(22), 0, "gourd"),
          (j(-86), j(32), 0, "jack"), (j(-104), j(32), 0, "plain")]
    # entrance: skeleton slumped against a bale, a lantern at its feet, candles
    P += [(j(-38), 44, 0, "bale"), (-22, 50, 0, "skeleton"), (j(-2), 56, 0, "jack"),
          (j(8), 52, 0, "candle"), (j(13), 56, 0, "candle"), (j(-50), 55, 0, "candle")]
    # right corner: pumpkin patch with bones in the leaves
    P += [(j(102), j(24), 0, "leaves"), (j(92), j(18), 0, "white"), (j(112), j(16), 0, "jack"),
          (j(106), j(30), 0, "plain"), (j(88), j(30), 0, "gourd"), (j(98), j(34), 0, "gourd"),
          (j(124), j(30), 0, "pile")]
    # a lantern either side of the doorway
    P += [(j(-66), j(40), 0, "jack"), (j(62), j(40), 0, "jack"), (j(54), j(44), 0, "candle")]
    return [(dx * flip, dy, lift, kind) for dx, dy, lift, kind in P]


def _render(kind, rng, flip):
    if kind == "bale":
        return harvest.hay_bale(rng)
    if kind == "leaves":
        return harvest.leaves(rng) + (0,)
    if kind == "candle":
        return harvest.candle(rng) + (0,)
    if kind == "skeleton":
        return harvest.skeleton_sitting(rng, flip) + (0,)
    if kind == "pile":
        return harvest.bone_pile(rng) + (0,)
    return harvest.pumpkin(rng, kind) + (0,)


def decorate(frame, variant=0):
    """Paint the display into an SLD frame in place. Only draws inside the existing box,
    so the frame's masks and outline data stay the right size."""
    rng = np.random.default_rng(1000 + variant)
    flip = -1 if variant % 2 else 1
    hx, hy = frame["hotspot"]
    main = frame["layers"]["main"]
    x1, y1, _, _ = main["rect"]
    px = main["pixels"]
    pc = frame["layers"].get("playercolor")
    sh = frame["layers"].get("shadow")

    rendered = []
    bale_top = 0
    for dx, dy, lift, kind in _layout(rng, flip):
        rgba, shadow, (phx, phy), top = _render(kind, rng, flip)
        if kind == "bale":
            bale_top = top
        if lift == "BALE":
            lift, shadow = bale_top, np.zeros_like(shadow)   # sits on the bale: no ground shadow
        # ground decals first, then everything else back to front
        order = (0 if kind == "leaves" else 1, dy + (0.5 if lift else 0))
        rendered.append((order, dx, dy, lift, rgba, shadow, phx, phy))
    rendered.sort(key=lambda r: r[0])

    for _, dx, dy, lift, rgba, shadow, phx, phy in rendered:
        vis = (rgba[..., 3] > 127) | (shadow > 0)
        if not vis.any():
            continue
        ys, xs = np.nonzero(vis)
        t, b, l, r = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rgba, shadow, phx, phy = rgba[t:b, l:r], shadow[t:b, l:r], phx - l, phy - t
        h, w = rgba.shape[:2]
        ox = int(round(hx + dx - phx - x1))
        oy = int(round(hy + dy - lift - phy - y1))
        # nudge back inside the box rather than resizing it
        oy = min(oy, px.shape[0] - h - 1)
        ox = min(max(ox, 0), px.shape[1] - w)
        if oy < 0:
            continue
        m = rgba[..., 3] > 127
        px[oy:oy + h, ox:ox + w][m] = rgba[m]
        if pc is not None:
            pc["pixels"][oy:oy + h, ox:ox + w][m] = 0
        if sh is not None and shadow.any():
            sx1, sy1, _, _ = sh["rect"]
            sox, soy = ox + x1 - sx1, oy + y1 - sy1
            sp = sh["pixels"]
            if 0 <= sox and 0 <= soy and sox + w <= sp.shape[1] and soy + h <= sp.shape[0]:
                region = sp[soy:soy + h, sox:sox + w]
                np.maximum(region, shadow, out=region)
    return frame
