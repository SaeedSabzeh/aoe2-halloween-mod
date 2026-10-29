"""Small human skeleton props for Town Centers: one sitting slumped, one bone pile."""
import numpy as np
import render3d
from render3d import Scene

# warm, slightly dirty bone so it sits in the game's palette instead of glowing white
BONE = (214, 202, 172)
BONE_SHADE = (176, 162, 130)
SOCKET = (46, 34, 28)
OUTLINE = (72, 58, 44)


def sitting(mirror=1.0, scale=2.0):
    """Skeleton sitting on the ground leaning back, head slumped forward.
    Local x is where it faces; it leans back towards -x (against a pumpkin)."""
    s = Scene()
    k = scale
    P = lambda x, y, z: (x * k, y * k * mirror, z * k)
    R = lambda r: r * k * 0.75  # thinner than scale so gaps between bones survive
    s.capsule(P(0, 1.5, 2.3), P(0, -1.5, 2.3), R(1.0), BONE_SHADE)            # pelvis
    s.capsule(P(-0.6, 0, 2.8), P(-2.0, 0, 8.6), R(0.6), BONE)                  # spine, leaning back
    for i, z in enumerate((4.8, 6.4, 8.0)):                                    # ribs
        for side in (1, -1):
            s.capsule(P(-1.6 + i * -0.2, 0, z), P(-0.4 + i * -0.2, side * 2.0, z - 0.6), R(0.42), BONE)
            s.capsule(P(-0.4 + i * -0.2, side * 2.0, z - 0.6), P(0.4 + i * -0.2, side * 0.6, z - 1.2), R(0.38), BONE)
    s.capsule(P(-2.0, 2.3, 8.6), P(-2.0, -2.3, 8.6), R(0.55), BONE_SHADE)       # collarbones
    # skull slumped forward and to one side
    s.sphere(P(-1.0, 0.6, 11.2), 2.3 * k * 0.8, BONE)
    s.sphere(P(0.4, 0.6, 9.9), R(1.1), BONE_SHADE)                             # jaw
    for side in (1, -1):
        s.sphere(P(0.6, 0.6 + side * 0.8, 11.3), 0.75 * k * 0.8, SOCKET)
    # legs: one knee up, one stretched out
    s.capsule(P(0, 1.3, 2.0), P(4.0, 1.7, 5.0), R(0.7), BONE)
    s.capsule(P(4.0, 1.7, 5.0), P(6.6, 2.0, 0.6), R(0.6), BONE)
    s.capsule(P(0, -1.3, 2.0), P(4.6, -1.8, 1.0), R(0.7), BONE)
    s.capsule(P(4.6, -1.8, 1.0), P(8.6, -2.0, 0.7), R(0.6), BONE)
    for fx, fy in ((6.6, 2.0), (8.6, -2.0)):
        s.capsule(P(fx, fy, 0.5), P(fx + 1.2, fy * 1.05, 0.4), R(0.5), BONE_SHADE)  # feet
    # arms hanging, hands resting on the ground
    for side, elbow, hand in ((1, (-0.6, 3.2, 4.8), (1.6, 3.4, 0.6)), (-1, (-0.2, -3.0, 5.2), (2.4, -2.6, 2.6))):
        s.capsule(P(-2.0, side * 2.4, 8.4), P(*elbow), R(0.5), BONE)
        s.capsule(P(*elbow), P(*hand), R(0.45), BONE)
    return s


def bone_pile(seed=0, scale=1.9):
    """A skull lying on its side with a few scattered long bones."""
    rng = np.random.default_rng(seed)
    s = Scene()
    k = scale
    P = lambda x, y, z: (x * k, y * k, z * k)
    for i in range(4):
        a = i * np.pi / 4 + rng.uniform(-0.3, 0.3)   # criss-crossed, not all parallel
        cx, cy = rng.uniform(-4, 2), rng.uniform(-3.5, 3.5)
        dx, dy = np.cos(a) * 3.0, np.sin(a) * 3.0
        s.capsule(P(cx - dx, cy - dy, 0.6), P(cx + dx, cy + dy, 0.6), 0.42 * k, BONE)
        for e in (-1, 1):
            s.sphere(P(cx + e * dx * 1.05, cy + e * dy * 1.05, 0.7), 0.65 * k, BONE_SHADE)
    s.sphere(P(1.5, -0.5, 1.8), 1.9 * k, BONE)
    s.sphere(P(2.6, -0.2, 1.0), 1.0 * k, BONE_SHADE)
    s.sphere(P(3.1, -1.0, 2.1), 0.5 * k, SOCKET)
    s.sphere(P(3.1, 0.4, 2.1), 0.5 * k, SOCKET)
    return s


def render(kind, facing, seed=0, mirror=1.0):
    scene = sitting(mirror) if kind == "sitting" else bone_pile(seed)
    rgba, _, sh, hs = render3d.render(scene, facing, (52, 48), ground=12, shadow_alpha=120, outline=OUTLINE)
    return rgba, sh, hs
