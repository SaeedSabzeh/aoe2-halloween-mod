"""Halloween creatures built for render3d: spider sheep and skeleton boar.

Each builder takes:
  phase     0..1 position in the walk/attack cycle
  pose      "idle", "walk", "attack", "dead"
  amount    0..1 for "dead": how much carcass is left (1 = just died)
  collapse  0..1 for death animations: blends the living pose into the dead one
"""
import numpy as np
from render3d import Scene

TAU = 2 * np.pi

WOOL = (64, 56, 72)
WOOL_SHADE = (42, 36, 48)
LEG_BAND = (232, 118, 28)
SPIDER_BLACK = (34, 30, 36)
SPIDER_LEG = (46, 40, 48)
EYE_RED = (255, 40, 30)
FANG = (235, 228, 210)

BONE = (232, 224, 202)
BONE_DARK = (190, 178, 150)
SOCKET = (24, 18, 18)
EYE_GLOW = (120, 255, 90)


def _lerp(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


# ----------------------------------------------------------------- spider sheep

def spider_sheep(phase=0.0, pose="idle", amount=1.0, collapse=0.0):
    s = Scene()
    flip = collapse if pose == "dead" else 0.0
    bob = 0.6 * np.sin(phase * TAU * 2) if pose == "walk" else 0.0
    if pose == "idle":
        bob = 0.3 * np.sin(phase * TAU)
    body_z = _lerp(9.0 + bob, 5.0, flip)

    # wool body made of fluffy blobs; shrinks as it's eaten
    rng = np.random.default_rng(3)
    blobs = [(0, 0, 0, 6.2)] + [
        (rng.uniform(-6, 5), rng.uniform(-3.5, 3.5), rng.uniform(-2, 3), rng.uniform(2.8, 4.0))
        for _ in range(14)]
    keep = max(1, int(round(len(blobs) * (amount if pose == "dead" else 1.0))))
    for i, (bx, by, bz, br) in enumerate(blobs[:keep]):
        col = WOOL if bz > 0 else WOOL_SHADE
        s.sphere((bx, by, body_z + bz * (1 - 2 * flip)), br, col)
    # player colour hourglass on the back
    s.sphere((-1.5, 0, body_z + 5.6 * (1 - 2 * flip)), 1.6, (200, 30, 30), pc=1.0)
    s.sphere((1.0, 0, body_z + 5.8 * (1 - 2 * flip)), 1.6, (200, 30, 30), pc=1.0)

    # head + eyes + fangs
    head_x = 8.5 + (1.5 * np.sin(phase * TAU) if pose == "attack" else 0)
    head_z = body_z - 0.5 - (2.0 * np.sin(phase * TAU) if pose == "attack" else 0)
    s.sphere((head_x, 0, head_z), 3.6, SPIDER_BLACK)
    if pose != "dead" or amount > 0.3:
        for ex, ey, ez, er in [(2.6, 1.0, 1.6, 0.85), (2.6, -1.0, 1.6, 0.85),
                               (2.2, 2.0, 2.2, 0.6), (2.2, -2.0, 2.2, 0.6),
                               (3.0, 0.5, 0.6, 0.5), (3.0, -0.5, 0.6, 0.5)]:
            s.sphere((head_x + ex, ey, head_z + ez * (1 - 2 * flip)), er, EYE_RED)
        fang_open = 0.8 * np.sin(phase * TAU) if pose == "attack" else 0
        for sy in (1, -1):
            s.capsule((head_x + 2.8, sy * 1.0, head_z - 1.8 * (1 - 2 * flip)),
                      (head_x + 3.6, sy * (0.6 + fang_open), head_z - 4.0 * (1 - 2 * flip)),
                      0.55, FANG, r1=0.25)

    # eight legs: alternating tetrapod gait
    for i, lx in enumerate((5.0, 2.0, -1.5, -4.5)):
        for side in (1, -1):
            group = (i + (side > 0)) % 2
            ph = phase * TAU + group * np.pi
            stride = 4.5 * np.sin(ph) if pose == "walk" else 0.0
            lift = max(0.0, np.cos(ph)) * 3.5 if pose == "walk" else 0.0
            if pose == "attack" and i == 0:
                lift, stride = 4.0 * max(0, np.sin(phase * TAU)), 2.0
            # fan the legs out like a real spider: front pair reaches forward,
            # back pair reaches back, so all eight feet land in separate spots
            ang = np.radians((38, 72, 108, 145)[i])
            reach = (17.0, 15.5, 15.5, 17.5)[i]
            hip = (lx * 0.6, side * 3.5, body_z + 1.0)
            fx, fy = np.cos(ang) * reach, np.sin(ang) * reach
            knee = (fx * 0.55 + stride * 0.4, side * fy * 0.55, body_z + 7.5 + lift * 0.5)
            foot = (fx + stride, side * fy, lift)
            if flip > 0:
                # dead spider: legs curl up over the belly
                knee_d = (lx * 0.8, side * 6.5, body_z + 9.0)
                foot_d = (lx * 0.5, side * 3.0, body_z + 11.5)
                knee = _lerp(knee, knee_d, flip)
                foot = _lerp(foot, foot_d, flip)
            s.capsule(hip, knee, 0.8, SPIDER_LEG, r1=0.65)
            s.capsule(knee, foot, 0.65, SPIDER_LEG, r1=0.4)
            # tarantula-style orange bands at the knee and shin, orange foot tips
            s.sphere(knee, 0.95, LEG_BAND)
            s.sphere(_lerp(knee, foot, 0.5), 0.7, LEG_BAND)
            s.sphere(foot, 0.75, LEG_BAND)
    return s


# ----------------------------------------------------------------- skeleton boar

def skeleton_boar(phase=0.0, pose="idle", amount=1.0, collapse=0.0, scale=1.25):
    s = Scene()
    c = collapse if pose == "dead" else 0.0

    def jitter(x, y, z, k):
        # stable pseudo-random offset per bone, so the pile doesn't flicker between frames
        v = np.sin(x * 12.9898 + y * 78.233 + z * 37.719 + k * 4.1) * 43758.5453
        return (v - np.floor(v)) * 2 - 1

    def P(x, y, z, scatter=1.0):
        # living point -> collapsed pile of bones on the ground
        pile = (x * 1.15 + jitter(x, y, z, 0) * 2 * scatter, y * 1.6 + jitter(x, y, z, 1) * 3 * scatter,
                1.2 + abs(z) * 0.06)
        p = _lerp((x, y, z), pile, c)
        return tuple(np.asarray(p) * scale)

    def R(r):
        return r * scale

    walk = pose == "walk"
    lunge = np.sin(phase * TAU) if pose == "attack" else 0.0
    bob = 0.6 * abs(np.sin(phase * TAU)) if walk else 0.25 * np.sin(phase * TAU)

    # spine
    spine = [(-13, 0, 15.5), (-8, 0, 17.0), (-2, 0, 18.0), (4, 0, 18.5), (10, 0, 17.5)]
    spine = [(x + lunge * 1.5, y, z + bob) for x, y, z in spine]
    for a, b in zip(spine, spine[1:]):
        s.capsule(P(*a), P(*b), R(1.2), BONE)
    for x, y, z in spine:
        s.sphere(P(x, y, z + 1.2), R(1.0), BONE_DARK)  # vertebra knobs
    # tail
    s.capsule(P(*spine[0]), P(-17, 0, 12 + bob), R(0.7), BONE, r1=R(0.4))

    # ribcage: hidden as the carcass is picked clean
    n_ribs = 6 if pose != "dead" else max(0, int(round(6 * amount)))
    for k in range(n_ribs):
        x = -6 + k * 2.6 + lunge * 1.5
        top_z = 17.8 + bob
        for side in (1, -1):
            mid = (x + 0.5, side * 6.0, 12.5 + bob)
            low = (x + 1.2, side * 3.0, 7.5 + bob)
            s.capsule(P(x, side * 1.0, top_z), P(*mid), R(0.65), BONE)
            s.capsule(P(*mid), P(*low), R(0.6), BONE, r1=R(0.45))

    # pelvis and shoulder blades
    s.capsule(P(-12 + lunge, 3.5, 14.5 + bob), P(-12 + lunge, -3.5, 14.5 + bob), R(1.5), BONE_DARK)
    for side in (1, -1):
        s.capsule(P(8 + lunge, side * 3.5, 17 + bob), P(6.5 + lunge, side * 4.5, 12 + bob), R(1.3), BONE, r1=R(0.9))

    # legs: trot with diagonal pairs
    for lx, side, pair in ((8, 1, 0), (8, -1, 1), (-11, 1, 1), (-11, -1, 0)):
        ph = phase * TAU + pair * np.pi
        swing = 5.0 * np.sin(ph) if walk else 0.0
        lift = max(0.0, np.cos(ph)) * 3.0 if walk else 0.0
        lx2 = lx + lunge * (1.0 if lx > 0 else 0.4)
        hip = (lx2, side * 3.8, 14.5 + bob)
        knee = (lx2 + swing * 0.5 + (1.2 if lx > 0 else -1.5), side * 4.0, 7.5 + lift)
        hoof = (lx2 + swing, side * 4.0, 0.8 + lift)
        s.capsule(P(*hip), P(*knee), R(1.1), BONE, r1=R(0.85))
        s.capsule(P(*knee), P(*hoof), R(0.85), BONE, r1=R(0.65))
        s.sphere(P(*knee), R(1.0), BONE_DARK)
        s.sphere(P(hoof[0] + 0.6, hoof[1], hoof[2]), R(0.9), BONE_DARK)

    # skull with tusks and glowing eyes; dips forward when attacking
    hx = 15.5 + lunge * 4.0
    hz = 15.0 + bob - lunge * 4.0
    s.capsule(P(13 + lunge * 2, 0, 16.5 + bob), P(hx, 0, hz), R(1.3), BONE)  # neck
    s.sphere(P(hx + 1.0, 0, hz + 0.5, 0.3), R(4.2), BONE)
    s.capsule(P(hx + 3.0, 0, hz - 0.5, 0.3), P(hx + 9.0, 0, hz - 3.0, 0.3), R(2.7), BONE, r1=R(1.8))
    s.sphere(P(hx + 9.6, 0, hz - 3.2, 0.3), R(1.6), BONE_DARK)  # snout
    for side in (1, -1):
        s.sphere(P(hx + 3.0, side * 2.6, hz + 1.6, 0.3), R(1.25), SOCKET)
        if pose != "dead":
            s.sphere(P(hx + 3.5, side * 2.9, hz + 1.7, 0.3), R(0.55), EYE_GLOW)
        # curved tusks
        t0 = (hx + 7.0, side * 1.9, hz - 3.5)
        t1 = (hx + 8.8, side * 3.2, hz - 1.2)
        t2 = (hx + 8.4, side * 3.0, hz + 1.2)
        s.capsule(P(*t0, 0.3), P(*t1, 0.3), R(0.75), FANG, r1=R(0.55))
        s.capsule(P(*t1, 0.3), P(*t2, 0.3), R(0.55), FANG, r1=R(0.3))
        # lower jaw
        s.capsule(P(hx + 2.0, side * 1.6, hz - 2.6, 0.3), P(hx + 7.5, side * 1.2, hz - 4.6 - lunge * 1.5, 0.3),
                  R(0.9), BONE_DARK)
    return s
