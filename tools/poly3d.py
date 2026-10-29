"""Painter's-algorithm renderer for flat-faced objects (boxes, discs, scooters).

Same world space and camera as render3d: x forward, y left, z up, viewed from
AoE2's 2:1 isometric camera. Output matches render3d.render.
"""
import numpy as np
from PIL import Image, ImageDraw

SS = 3
ELEV = np.radians(30)
VIEW = np.array([0.0, -np.cos(ELEV), np.sin(ELEV)])     # towards the camera, after facing
LIGHT = np.array([-0.45, 0.55, 1.0]) / np.linalg.norm([-0.45, 0.55, 1.0])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


class Scene:
    def __init__(self):
        self.faces = []   # (points Nx3, color, pc, two_sided, bias)
        self.M = np.eye(3)
        self.T = np.zeros(3)
        self._stack = []

    # -- transform stack: points are mapped by M @ p + T
    def push(self, R=None, t=(0, 0, 0)):
        self._stack.append((self.M.copy(), self.T.copy()))
        t = np.asarray(t, float)
        self.T = self.M @ t + self.T
        if R is not None:
            self.M = self.M @ R

    def pop(self):
        self.M, self.T = self._stack.pop()

    def poly(self, pts, color, pc=0.0, two_sided=False, bias=0.0):
        pts = np.asarray(pts, float) @ self.M.T + self.T
        self.faces.append((pts, np.asarray(color, float), pc, two_sided, bias))

    def box(self, size, color, top=None, pc_sides=0.0, pc_top=0.0, side_color=None):
        """Axis-aligned box from (0,0,0) to size in the current frame (bottom-centred on x/y)."""
        sx, sy, sz = size
        x0, x1, y0, y1, z0, z1 = -sx / 2, sx / 2, -sy / 2, sy / 2, 0, sz
        side = side_color if side_color is not None else color
        self.poly([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], top if top is not None else color, pc_top)
        self.poly([(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)], color)
        self.poly([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], side, pc_sides)
        self.poly([(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)], side, pc_sides)
        self.poly([(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)], side, pc_sides)
        self.poly([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], side, pc_sides)

    def disc(self, center, r, color, segments=20, a0=0.0, a1=2 * np.pi, pc=0.0, bias=0.0, sy=1.0):
        """Flat disc (or wedge from a0 to a1) lying in the current frame's x/y plane."""
        cx, cy, cz = center
        angs = np.linspace(a0, a1, max(3, int(segments * (a1 - a0) / (2 * np.pi)) + 1))
        ring = [(cx + np.cos(a) * r, cy + np.sin(a) * r * sy, cz) for a in angs]
        pts = ring if a1 - a0 >= 2 * np.pi - 1e-6 else [(cx, cy, cz)] + ring
        self.poly(pts, color, pc, two_sided=True, bias=bias)


def render(scene, facing, size, ground=None, shadow_alpha=150, outline=(24, 18, 14)):
    w, h = size
    W, H = w * SS, h * SS
    ox = W / 2
    oy = H - (ground if ground is not None else h * 0.25) * SS
    R = rot_z(facing)
    items = []
    for k, (pts, color, pc, two_sided, bias) in enumerate(scene.faces):
        p = pts @ R.T
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9 and len(p) > 3:
            n = np.cross(p[2] - p[0], p[3] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        facing_cam = n @ VIEW
        if facing_cam < 0:
            if not two_sided:
                continue
            n = -n
        light = 0.62 + 0.38 * max(0.0, n @ LIGHT)
        sx = p[:, 0]
        sy = -p[:, 1] * np.sin(ELEV) - p[:, 2] * np.cos(ELEV)
        depth = (-p[:, 1] * np.cos(ELEV) + p[:, 2] * np.sin(ELEV)).mean() + bias
        items.append((depth, k, sx, sy, color * light, pc, p))
    items.sort(key=lambda it: (it[0], it[1]))

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pcimg = Image.new("L", (W, H), 0)
    shimg = Image.new("L", (W, H), 0)
    d, dp, ds = ImageDraw.Draw(img), ImageDraw.Draw(pcimg), ImageDraw.Draw(shimg)
    for depth, k, sx, sy, col, pc, p in items:
        poly = [(ox + x * SS, oy + y * SS) for x, y in zip(sx, sy)]
        c = tuple(int(v) for v in np.clip(col, 0, 255)) + (255,)
        d.polygon(poly, fill=c)
        dp.polygon(poly, fill=int(pc * 255))
        # shadow: drop onto the ground, pushed down-right by height
        g = [(ox + (x + z * 0.45) * SS, oy + (-y * np.sin(ELEV) + z * 0.12) * SS) for x, y, z in p]
        ds.polygon(g, fill=255)

    rgb = np.asarray(img, dtype=np.float64)[..., :3]
    alpha = (np.asarray(img)[..., 3] > 0).astype(float)
    pcm = np.asarray(pcimg, dtype=np.float64) / 255
    shadow = np.asarray(shimg, dtype=np.float64) / 255
    rgb = rgb.copy()
    if outline is not None:
        ring = np.zeros_like(alpha, bool)
        for ddy in range(-SS, SS + 1):
            for ddx in range(-SS, SS + 1):
                if ddx * ddx + ddy * ddy <= SS * SS:
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
