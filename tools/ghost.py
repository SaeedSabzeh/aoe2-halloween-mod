"""Turn the game's own animal sprites into ghosts (keeps the original animation)."""
import numpy as np

# 4x4 ordered dither: sprites only have on/off alpha, so see-through is faked with a
# fine pattern that reads as translucent at game scale
BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16

DEEP = np.array([96, 140, 196])
PALE = np.array([226, 240, 255])
RIM = np.array([238, 250, 255])
FLOAT = 4   # px the ghost hovers above the ground


def ghostify(frame, opacity=1.0, float_px=FLOAT, wisp=True):
    """frame: an SLD frame dict from sld.read; returns a new frame dict.
    opacity 0..1 scales how solid the ghost is (used to fade it out on death)."""
    if "main" not in frame["layers"]:
        return dict(width=frame["width"], height=frame["height"], hotspot=frame["hotspot"], unk=frame["unk"], layers={})
    m = frame["layers"]["main"]
    x1, y1, x2, y2 = m["rect"]
    px = m["pixels"].astype(np.float64)
    a = px[..., 3] > 127
    out = np.zeros(px.shape, np.uint8)
    if a.any():
        lum = px[..., :3] @ np.array([0.3, 0.59, 0.11]) / 255
        lo, hi = np.percentile(lum[a], 5), np.percentile(lum[a], 98)
        t = np.clip((lum - lo) / max(hi - lo, 1e-6), 0, 1)[..., None]
        col = DEEP * (1 - t) + PALE * t
        # glowing rim around the silhouette
        inner = a & np.roll(a, 1, 0) & np.roll(a, -1, 0) & np.roll(a, 1, 1) & np.roll(a, -1, 1)
        rim = a & ~inner
        col[rim] = col[rim] * 0.45 + RIM * 0.55   # soft glow, not a hard outline
        # solid at the top, wisping away towards the hooves
        ys = np.nonzero(a)[0]
        top, bottom = ys.min(), ys.max()
        rows = np.arange(px.shape[0])[:, None]
        # body stays nearly solid; only the lower third (legs) wisps away
        fade = np.clip((bottom - rows) / max((bottom - top) * 0.4, 1), 0, 1) if wisp else np.ones_like(rows, float)
        solid = opacity * (0.3 + 0.68 * fade)
        yy, xx = np.mgrid[:px.shape[0], :px.shape[1]]
        thresh = BAYER[(yy + y1) % 4, (xx + x1) % 4]   # canvas-aligned so it doesn't crawl
        keep = a & (solid > thresh)
        out[..., :3] = np.clip(col, 0, 255).astype(np.uint8)
        out[..., 3] = np.where(keep, 255, 0)
        out[~keep, :3] = 0
    # hover: shift the sprite up; ghosts cast no shadow
    y_up = min(float_px, y1 - (y1 % 4))
    y_up -= y_up % 4
    layers = {"main": dict(rect=(x1, y1 - y_up, x2, y2 - y_up), pixels=out)}
    return dict(width=frame["width"], height=frame["height"], hotspot=frame["hotspot"], unk=frame["unk"], layers=layers)
