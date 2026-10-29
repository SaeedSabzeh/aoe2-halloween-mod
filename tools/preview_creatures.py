import sys, numpy as np
sys.path.insert(0, ".")
import render3d, creatures
from PIL import Image

def comp(rgba, sh, pc_mask, pc_col=(40, 80, 230)):
    bg = np.zeros((rgba.shape[0], rgba.shape[1], 3)); bg[:] = [110, 140, 70]
    bg *= (1 - sh[..., None] / 255 * 0.8)
    col = rgba[..., :3].astype(float)
    t = pc_mask[..., None] / 255
    col = col * (1 - t) + np.array(pc_col) * t
    a = rgba[..., 3:4] / 255
    return (bg * (1 - a) + col * a).astype(np.uint8)

def sheet(builder, size, rows, out):
    lines = []
    for row in rows:
        tiles = []
        for facing, kw in row:
            rgba, pc, sh, hs = render3d.render(builder(**kw), facing, size)
            tiles.append(np.pad(comp(rgba, sh, pc), ((2, 2), (2, 2), (0, 0))))
        lines.append(np.concatenate(tiles, axis=1))
    img = np.concatenate(lines, axis=0)
    Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4), Image.NEAREST).save(out)

dirs = [i * np.pi / 4 for i in range(8)]
sheet(creatures.spider_sheep, (48, 40), [
    [(d, dict(pose="idle")) for d in dirs],
    [(np.pi / 4 * 7, dict(pose="walk", phase=p)) for p in np.linspace(0, 1, 8, endpoint=False)],
    [(np.pi / 4 * 7, dict(pose="dead", collapse=c)) for c in np.linspace(0, 1, 5)] +
    [(np.pi / 4 * 7, dict(pose="dead", collapse=1, amount=a)) for a in (0.6, 0.3, 0.1)],
], "../preview_spider.png")
sheet(creatures.skeleton_boar, (72, 56), [
    [(d, dict(pose="idle")) for d in dirs],
    [(np.pi / 4 * 7, dict(pose="walk", phase=p)) for p in np.linspace(0, 1, 8, endpoint=False)],
    [(np.pi / 4 * 7, dict(pose="attack", phase=p)) for p in np.linspace(0, 0.5, 4)] +
    [(np.pi / 4 * 7, dict(pose="dead", collapse=c)) for c in (0.3, 0.7, 1.0)] +
    [(np.pi / 4 * 7, dict(pose="dead", collapse=1, amount=0.2))],
], "../preview_skeleton.png")
