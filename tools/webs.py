"""Draw spider webs over existing tree sprites.

Works in canvas coordinates of an SLD frame. Webs can extend past the tree's
own box; `span=True` adds big webs strung over to where neighbouring forest
trees stand, so dense jungle looks webbed across several trees.
"""
import numpy as np

SILK = np.array([236, 236, 242])
SILK_DIM = np.array([196, 198, 206])

# screen offsets of the neighbouring tiles in front of this one (AoE2 2:1 tiles)
FRONT_NEIGHBOURS = [(48, 24), (-48, 24), (0, 48)]


class Layer:
    """RGBA buffer positioned on the frame canvas; grows when drawn outside."""

    def __init__(self, pixels, rect):
        self.px = pixels.copy()
        self.x1, self.y1 = rect[0], rect[1]

    def grow(self, x0, y0, x1, y1):
        # layer boxes are stored unsigned, so they can't start left of / above the canvas
        x0, y0 = max(0, x0), max(0, y0)
        h, w = self.px.shape[:2]
        left = max(0, self.x1 - x0)
        top = max(0, self.y1 - y0)
        right = max(0, x1 - (self.x1 + w))
        bottom = max(0, y1 - (self.y1 + h))
        # keep everything on the 4px block grid
        left += (-left) % 4; top += (-top) % 4
        right += (-(w + left + right)) % 4
        bottom += (-(h + top + bottom)) % 4
        if left or top or right or bottom:
            self.px = np.pad(self.px, ((top, bottom), (left, right), (0, 0)))
            self.x1 -= left
            self.y1 -= top

    def rect(self):
        h, w = self.px.shape[:2]
        return (self.x1, self.y1, self.x1 + w, self.y1 + h)

    def plot(self, x, y, strength=1.0):
        X, Y = int(round(x)) - self.x1, int(round(y)) - self.y1
        h, w = self.px.shape[:2]
        if not (0 <= X < w and 0 <= Y < h):
            return
        p = self.px[Y, X]
        if p[3] > 127:
            # silk over foliage: blend so the tree shows through
            p[:3] = np.clip(p[:3] * (1 - 0.7 * strength) + SILK * 0.7 * strength, 0, 255)
        else:
            p[:3] = SILK_DIM
            p[3] = 255

    def line(self, a, b, sag=0.0, strength=1.0):
        (ax, ay), (bx, by) = a, b
        n = int(max(abs(bx - ax), abs(by - ay))) + 1
        for i in range(n + 1):
            t = i / max(1, n)
            x = ax + (bx - ax) * t
            y = ay + (by - ay) * t + sag * 4 * t * (1 - t)
            self.plot(x, y, strength)


def orb(layer, cx, cy, r, rng, spokes=None, rings=None, squash=1.0):
    spokes = spokes or rng.integers(8, 12)
    rings = rings or max(3, int(r / 3.5))
    angs = np.sort(rng.uniform(0, 2 * np.pi, spokes) * 0.35 + np.linspace(0, 2 * np.pi, spokes, endpoint=False))
    lens = r * rng.uniform(0.8, 1.05, spokes)
    tips = [(cx + np.cos(a) * l, cy + np.sin(a) * l * squash) for a, l in zip(angs, lens)]
    layer.grow(int(cx - r - 2), int(cy - r - 2), int(cx + r + 3), int(cy + r + 3))
    for t in tips:
        layer.line((cx, cy), t)
    for k in range(1, rings + 1):
        f = k / (rings + 0.5)
        pts = [(cx + (tx - cx) * f, cy + (ty - cy) * f) for tx, ty in tips]
        for p, q in zip(pts, pts[1:] + pts[:1]):
            layer.line(p, q, sag=0.6 * f, strength=0.85)
    return tips


def opaque_points(layer):
    ys, xs = np.nonzero(layer.px[..., 3] > 127)
    return xs + layer.x1, ys + layer.y1


def _on_edge(poly, rng):
    """Random point on the outline of a polygon."""
    i = rng.integers(len(poly))
    a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
    return tuple(a + (b - a) * rng.random())


def gauze(layer, poly, rng, strands=30, strength=0.3, sag=3.0):
    """A cobweb sheet: many faint strands criss-crossing a polygon. Overlapping
    sheets build up, so the more layers a spot has, the whiter it gets."""
    xs, ys = zip(*poly)
    layer.grow(int(min(xs)) - 2, int(min(ys)) - 2, int(max(xs)) + 3, int(max(ys)) + int(sag) + 3)
    for _ in range(strands):
        a, b = _on_edge(poly, rng), _on_edge(poly, rng)
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 4:
            layer.line(a, b, sag=rng.uniform(0, sag), strength=strength)


def web_tree(layer, hotspot, seed, span=False):
    """Haunted-wood cobwebs over one tree frame (canvas coords)."""
    rng = np.random.default_rng(seed)
    xs, ys = opaque_points(layer)
    if len(xs) == 0:
        return layer
    top, bottom = ys.min(), ys.max()
    height = bottom - top
    hx, hy = hotspot

    def extent(y):
        row = xs[np.abs(ys - y) <= 1]
        return (row.min(), row.max()) if len(row) else (hx, hx)

    canopy_bottom = top + height * 0.85

    # overlapping gauze sheets draped over the canopy
    for _ in range(int(rng.integers(4, 7))):
        y_top = top + height * rng.uniform(0.0, 0.45)
        y_low = min(canopy_bottom, y_top + height * rng.uniform(0.3, 0.55))
        l0, r0 = extent(y_top)
        l1, r1 = extent(y_low)
        over = rng.uniform(1, 5)  # sheets hang a little past the branches
        poly = [((l0 + r0) / 2 + rng.uniform(-3, 3), y_top),
                (r1 + over, y_low + rng.uniform(-2, 4)),
                (l1 - over, y_low + rng.uniform(-2, 4))]
        if rng.random() < 0.5:  # some sheets only cover one side
            mid = ((l1 + r1) / 2, y_low)
            poly[1 if rng.random() < 0.5 else 2] = mid
        gauze(layer, poly, rng, strands=int(rng.integers(22, 36)), strength=rng.uniform(0.25, 0.4))

    # tattered strands hanging off the branches
    for _ in range(int(rng.integers(5, 9))):
        y = top + height * rng.uniform(0.2, 0.85)
        l, r = extent(y)
        x = l if rng.random() < 0.5 else r
        drop = rng.uniform(6, 22)
        layer.grow(int(x) - 4, int(y) - 2, int(x) + 5, int(y + drop) + 3)
        layer.line((x, y), (x + rng.uniform(-3, 3), y + drop), sag=rng.uniform(-1.5, 1.5), strength=0.55)

    # the odd orb web on top of the gauze
    if rng.random() < 0.35:
        y = top + height * rng.uniform(0.2, 0.6)
        l, r = extent(y)
        orb(layer, l + (r - l) * rng.uniform(0.3, 0.7), y, float(np.clip((r - l) * 0.3, 5, 10)), rng)

    # veils strung over to the trees in front. Only for jungle, which always grows in
    # dense forest; other trees can stand alone and would trail webs into empty air.
    for dx, dy in FRONT_NEIGHBOURS[:2]:
        if not span or rng.random() > 0.8:
            continue
        y_a = top + height * rng.uniform(0.25, 0.5)
        y_b = y_a + height * rng.uniform(0.2, 0.35)
        la, ra = extent(y_a)
        lb, rb = extent(y_b)
        own_a, own_b = (ra if dx > 0 else la, y_a), (rb if dx > 0 else lb, y_b)
        # the neighbour's matching edge, one tile over
        far_a = (own_a[0] + dx - (ra - la) * np.sign(dx), y_a + dy)
        far_b = (own_b[0] + dx - (rb - lb) * np.sign(dx), y_b + dy)
        gauze(layer, [own_a, far_a, far_b, own_b], rng, strands=int(rng.integers(14, 24)),
              strength=0.3, sag=6.0)
    return layer
