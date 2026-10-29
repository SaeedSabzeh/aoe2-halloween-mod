"""Read AoE2 DE .smx sprites (palette-indexed) into RGBA frames."""
import os
import struct
import numpy as np

import modkit
PALETTES = modkit.GAME_DIR + "resources/_common/palettes/"
# like .sld, real files number the frame bits from the lowest bit
F_MAIN, F_SHADOW, F_OUTLINE, F_8TO5 = 0x01, 0x02, 0x04, 0x08

_pal_cache = {}


def palette(number):
    if number not in _pal_cache:
        conf = {}
        for line in open(PALETTES + "palettes.conf", encoding="utf-8", errors="ignore"):
            line = line.strip()
            if line and not line.startswith("//") and "," in line:
                k, v = line.split(",", 1)
                conf[int(k)] = v.strip()
        # empty placeholder frames (e.g. removed stumps) use 255; fall back to the tree palette
        lines = open(PALETTES + conf.get(number, conf[30]), encoding="utf-8", errors="ignore").read().split()
        n = int(lines[2])
        vals = list(map(int, lines[3:]))
        per = len(vals) // n
        pal = np.array(vals[:n * per], dtype=np.uint8).reshape(n, per)[:, :3]
        _pal_cache[number] = pal
    return _pal_cache[number]


def _edges(data, pos, h):
    edges = [struct.unpack_from("<HH", data, pos + 4 * r) for r in range(h)]
    return edges, pos + 4 * h


def _main_layer(data, pos, w, h, compressed_8to5, pal):
    edges, pos = _edges(data, pos, h)
    cmd_len, px_len = struct.unpack_from("<II", data, pos); pos += 8
    cmds = data[pos:pos + cmd_len]; pos += cmd_len
    pix = data[pos:pos + px_len]; pos += px_len
    # decode the pixel stream into combined palette indices (section * 256 + index)
    if not compressed_8to5:
        chunks = np.frombuffer(pix[:len(pix) // 5 * 5], np.uint8).reshape(-1, 5)
        idx = chunks[:, :4].astype(np.int32)
        sec = np.stack([(chunks[:, 4] >> s) & 3 for s in (0, 2, 4, 6)], 1).astype(np.int32)
        colours = (idx + 256 * sec).reshape(-1)
    else:
        chunks = np.frombuffer(pix[:len(pix) // 5 * 5], np.uint8).reshape(-1, 5).astype(np.uint64)
        v = sum(chunks[:, k] << np.uint64(8 * k) for k in range(5))
        p0 = (v & np.uint64(0xFF)) + 256 * ((v >> np.uint64(14)) & np.uint64(3))
        p1 = ((v >> np.uint64(8)) & np.uint64(0x3F)) | (((v >> np.uint64(22)) & np.uint64(3)) << np.uint64(6))
        p1 = p1 + 256 * ((v >> np.uint64(20)) & np.uint64(3))
        colours = np.stack([p0, p1], 1).reshape(-1).astype(np.int32)
    out = np.zeros((h, w, 4), np.uint8)
    pc = np.zeros((h, w), np.uint8)
    ci = pi = 0
    for r in range(h):
        left, right = edges[r]
        if left == 0xFFFF:
            continue
        x = left
        while ci < len(cmds):
            c = cmds[ci]; ci += 1
            op, cnt = c & 3, (c >> 2) + 1
            if op == 3:
                break
            if op == 0:
                x += cnt
                continue
            for _ in range(cnt):
                if x < w:
                    k = colours[pi] % len(pal)
                    out[r, x, :3] = pal[k]
                    out[r, x, 3] = 255
                    if op == 2:
                        pc[r, x] = 255
                x += 1
                pi += 1
    return out, pc, pos


def _shadow_layer(data, pos, w, h):
    edges, pos = _edges(data, pos, h)
    n, = struct.unpack_from("<I", data, pos); pos += 4
    arr = data[pos:pos + n]; pos += n
    out = np.zeros((h, w), np.uint8)
    i = 0
    for r in range(h):
        left, right = edges[r]
        if left == 0xFFFF:
            continue
        x = left
        while i < len(arr):
            c = arr[i]; i += 1
            op, cnt = c & 3, (c >> 2) + 1
            if op == 3:
                break
            if op == 0:
                x += cnt
                continue
            for _ in range(cnt):
                if x < w:
                    out[r, x] = arr[i]
                i += 1
                x += 1
    return out, pos


def read(path):
    data = open(path, "rb").read()
    sig, ver, nframes = struct.unpack_from("<4shh", data, 0)
    assert sig == b"SMPX", sig
    pos = 32
    frames = []
    for _ in range(nframes):
        ftype, palnum, _usize = struct.unpack_from("<BBI", data, pos); pos += 6
        pal = palette(palnum)
        frame = dict(type=ftype, palette=palnum)
        for bit in (F_MAIN, F_SHADOW, F_OUTLINE):
            if not ftype & bit:
                continue
            w, h, hx, hy, length, _ = struct.unpack_from("<HHhhII", data, pos)
            body = pos + 16
            if bit == F_MAIN:
                px, pc, _ = _main_layer(data, body, w, h, bool(ftype & F_8TO5), pal)
                frame["main"] = dict(pixels=px, playercolor=pc, hotspot=(hx, hy))
            elif bit == F_SHADOW:
                sh, _ = _shadow_layer(data, body, w, h)
                frame["shadow"] = dict(pixels=sh, hotspot=(hx, hy))
            pos = body + length
        frames.append(frame)
    return frames
