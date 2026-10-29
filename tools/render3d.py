"""Tiny z-buffered renderer for creatures built from spheres and capsules.

World space: x forward, y left, z up (pixels). The model is spun by `facing`
(radians, 0 = facing screen-right, counter-clockwise on the ground) and viewed
from AoE2's 2:1 isometric camera (30 degrees above the ground).
"""
import numpy as np

SS = 3
ELEV = np.radians(30)
SUN = np.array([-0.55, -0.62, 0.56]) / np.linalg.norm([-0.55, -0.62, 0.56])


class Scene:
    def __init__(self):
        self.prims = []  # (kind, p0, p1, r0, r1, color, pc)

    def sphere(self, p, r, color, pc=0.0):
        self.prims.append(("s", np.asarray(p, float), None, r, r, np.asarray(color, float), pc))

    def capsule(self, p0, p1, r0, color, r1=None, pc=0.0):
        self.prims.append(("c", np.asarray(p0, float), np.asarray(p1, float), r0,
                           r0 if r1 is None else r1, np.asarray(color, float), pc))


def _project(p, facing):
    c, s = np.cos(facing), np.sin(facing)
    x, y, z = p
    # rotate on the ground, then map ground x/y to screen with 2:1 squash
    gx = x * c - y * s
    gy = x * s + y * c
    sx = gx
    sy = -gy * np.sin(ELEV) - z * np.cos(ELEV)
    depth = -gy * np.cos(ELEV) + z * np.sin(ELEV)  # larger = nearer the camera
    return sx, sy, depth


def render(scene, facing, size, ground=None, shadow_alpha=150, outline=(20, 16, 14), soft=False, noise=0.0, seed=0):
    """Render to (rgba HxWx4, playercolor HxW, shadow HxW, hotspot).

    size: (w, h) canvas in pixels, multiples of 4. The world origin (on the
    ground) lands at the hotspot, `ground` px above the bottom edge.
    """
    w, h = size
    W, H = w * SS, h * SS
    ox = W / 2
    oy = H - (ground if ground is not None else h * 0.25) * SS
    ys, xs = np.mgrid[:H, :W].astype(np.float64)
    zbuf = np.full((H, W), -1e9)
    rgb = np.zeros((H, W, 3))
    pcm = np.zeros((H, W))
    shadow = np.zeros((H, W))

    for kind, p0, p1, r0, r1, color, pc in scene.prims:
        a = np.array(_project(p0, facing))
        b = a if p1 is None else np.array(_project(p1, facing))
        ax, ay = ox + a[0] * SS, oy + a[1] * SS
        bx, by = ox + b[0] * SS, oy + b[1] * SS
        rmax = max(r0, r1) * SS
        x0, x1 = int(max(0, min(ax, bx) - rmax - 1)), int(min(W, max(ax, bx) + rmax + 2))
        y0, y1 = int(max(0, min(ay, by) - rmax - 1)), int(min(H, max(ay, by) + rmax + 2))
        if x0 < x1 and y0 < y1:
            X, Y = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = np.clip(((X - ax) * dx + (Y - ay) * dy) / L2, 0, 1) if L2 > 1e-9 else np.zeros_like(X)
            px, py = ax + t * dx, ay + t * dy
            r = (r0 + (r1 - r0) * t) * SS
            d = np.hypot(X - px, Y - py)
            inside = d <= r
            bulge = np.sqrt(np.clip(1 - (d / np.maximum(r, 1e-6)) ** 2, 0, 1))
            depth = (a[2] + t * (b[2] - a[2])) * SS + bulge * r
            win = inside & (depth > zbuf[y0:y1, x0:x1])
            # fake lighting from the upper left
            nx = (X - px) / np.maximum(r, 1e-6)
            ny = (Y - py) / np.maximum(r, 1e-6)
            if soft:
                # game-like lighting: sun from the upper left, soft falloff, faint highlight,
                # darker where things meet the ground and towards the silhouette edge
                diff = np.clip(nx * SUN[0] + ny * SUN[1] + bulge * SUN[2], 0, 1)
                light = 0.34 + 0.7 * diff + 0.12 * diff ** 10
                light *= 0.82 + 0.18 * np.clip(bulge * 3, 0, 1)
                light *= np.clip((oy - Y) / (5.0 * SS), 0.62, 1.0)
            else:
                light = 0.62 + 0.38 * np.clip(bulge * 0.7 - nx * 0.35 - ny * 0.45, 0, 1)
            sub = rgb[y0:y1, x0:x1]
            sub[win] = (color[None, :] * light[win][:, None])
            zbuf[y0:y1, x0:x1][win] = depth[win]
            pcm[y0:y1, x0:x1][win] = pc

        # shadow: drop the primitive onto the ground plane, pushed down-right
        def ground_pt(p):
            q = p.copy(); off = q[2]
            q[2] = 0
            sx, sy, _ = _project(q, facing)
            return ox + (sx + off * 0.45) * SS, oy + (sy + off * 0.12) * SS
        gax, gay = ground_pt(p0)
        gbx, gby = (gax, gay) if p1 is None else ground_pt(p1)
        x0, x1 = int(max(0, min(gax, gbx) - rmax - 1)), int(min(W, max(gax, gbx) + rmax + 2))
        y0, y1 = int(max(0, min(gay, gby) - rmax - 1)), int(min(H, max(gay, gby) + rmax + 2))
        if x0 < x1 and y0 < y1:
            X, Y = xs[y0:y1, x0:x1], ys[y0:y1, x0:x1]
            dx, dy = gbx - gax, gby - gay
            L2 = dx * dx + dy * dy
            t = np.clip(((X - gax) * dx + (Y - gay) * dy) / L2, 0, 1) if L2 > 1e-9 else np.zeros_like(X)
            r = (r0 + (r1 - r0) * t) * SS
            # squash vertically like a flat ellipse on the ground
            d = np.hypot(X - (gax + t * dx), (Y - (gay + t * dy)) * 1.6)
            shadow[y0:y1, x0:x1] = np.maximum(shadow[y0:y1, x0:x1], (d <= r).astype(float))

    alpha = (zbuf > -1e9).astype(float)
    if noise:
        # fine surface grain so props don't look airbrushed next to painted sprites
        g = np.random.default_rng(seed).normal(0, noise, (H // SS + 1, W // SS + 1))
        g = np.kron(g, np.ones((SS, SS)))[:H, :W]
        rgb *= (1 + g)[..., None]
    # dark outline one supersampled-pixel ring around the silhouette
    if outline is not None:
        ring = np.zeros_like(alpha, bool)
        k = SS
        for ddy in range(-k, k + 1):
            for ddx in range(-k, k + 1):
                if ddx * ddx + ddy * ddy <= k * k:
                    ring |= np.roll(np.roll(alpha > 0, ddy, 0), ddx, 1)
        ring &= alpha == 0
        rgb[ring] = outline
        alpha[ring] = 1

    def down(x):
        return x.reshape(h, SS, w, SS, *x.shape[2:]).mean(axis=(1, 3))
    a_d = down(alpha)
    rgb_d = down(rgb * alpha[..., None]) / np.maximum(a_d[..., None], 1e-6)
    pc_d = down(pcm * alpha) / np.maximum(a_d, 1e-6)
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.clip(rgb_d, 0, 255).astype(np.uint8)
    opaque = a_d >= 0.5
    out[..., 3] = np.where(opaque, 255, 0)
    pc_out = np.where(opaque, np.clip(pc_d * 255, 0, 255), 0).astype(np.uint8)
    sh = (down(shadow) * shadow_alpha).astype(np.uint8)
    sh[opaque] = 0
    return out, pc_out, sh, (w // 2, int(round(oy / SS)))
