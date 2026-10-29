"""Read and write AoE2 DE .sld sprite files (format v4).

Layers: main (BC1), shadow (BC4), unknown (bit 2, kept raw), damage (BC1),
playercolor (BC4). Pixel data is exposed as numpy arrays:
  main / damage -> HxWx4 uint8 RGBA, shadow / playercolor -> HxW uint8.
"""
import struct
import numpy as np
import outline

# The openage spec numbers bits MSB-first; in real files main is the lowest bit.
LAYER_MAIN, LAYER_SHADOW, LAYER_UNK, LAYER_DAMAGE, LAYER_PC = 0x01, 0x02, 0x04, 0x08, 0x10
FLAG_COPY_PREV = 0x80  # skipped blocks reuse the previous frame's block
# low flag bit as seen in the game's own files (main 0, everything else 1)
BASE_FLAG = {LAYER_MAIN: 0x00, LAYER_SHADOW: 0x01, LAYER_DAMAGE: 0x01, LAYER_PC: 0x01}
LAYER_ORDER = [LAYER_MAIN, LAYER_SHADOW, LAYER_UNK, LAYER_DAMAGE, LAYER_PC]
LAYER_NAMES = {LAYER_MAIN: "main", LAYER_SHADOW: "shadow", LAYER_UNK: "unknown",
               LAYER_DAMAGE: "damage", LAYER_PC: "playercolor"}


# ---------------------------------------------------------------- block codecs

def _565_to_rgb(c):
    r = (c >> 11) & 31; g = (c >> 5) & 63; b = c & 31
    return np.array([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], dtype=np.int32)


def _rgb_to_565(rgb):
    r, g, b = (int(round(v)) for v in np.clip(rgb, 0, 255))
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def bc1_decode(block):
    c0, c1, idx = struct.unpack("<HHI", block)
    p0, p1 = _565_to_rgb(c0), _565_to_rgb(c1)
    table = np.zeros((4, 4), dtype=np.int32)
    table[0, :3], table[1, :3] = p0, p1
    table[0:2, 3] = 255
    if c0 > c1:
        table[2, :3] = (2 * p0 + p1) // 3
        table[3, :3] = (p0 + 2 * p1) // 3
        table[2:4, 3] = 255
    else:
        table[2, :3] = (p0 + p1) // 2
        table[2, 3] = 255  # entry 3 stays transparent black
    out = np.zeros((16, 4), dtype=np.uint8)
    for i in range(16):
        out[i] = table[(idx >> (2 * i)) & 3]
    return out.reshape(4, 4, 4)


def bc1_encode(px):
    """px: 4x4x4 RGBA uint8 -> 8 bytes. Pixels with alpha < 128 become transparent."""
    px = px.reshape(16, 4).astype(np.float64)
    opaque = px[:, 3] >= 128
    if not opaque.any():
        return struct.pack("<HHI", 0, 0, 0xFFFFFFFF)
    cols = px[opaque, :3]
    # principal axis endpoints
    mean = cols.mean(axis=0)
    centered = cols - mean
    if len(cols) > 1 and np.abs(centered).sum() > 0:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        proj = centered @ axis
        e0, e1 = mean + axis * proj.max(), mean + axis * proj.min()
    else:
        e0 = e1 = mean
    c0, c1 = _rgb_to_565(e0), _rgb_to_565(e1)
    transparent = not opaque.all()
    if transparent:
        if c0 > c1:
            c0, c1 = c1, c0  # 3-colour + transparent mode needs c0 <= c1
    else:
        if c0 < c1:
            c0, c1 = c1, c0
        elif c0 == c1:
            # solid colour: 4-colour mode impossible, 3-colour mode works fine
            pass
    p0, p1 = _565_to_rgb(c0).astype(np.float64), _565_to_rgb(c1).astype(np.float64)
    if c0 > c1:
        table = [p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3]
        usable = [0, 1, 2, 3]
    else:
        table = [p0, p1, (p0 + p1) / 2]
        usable = [0, 1, 2]
    idx = 0
    for i in range(16):
        if not opaque[i]:
            sel = 3
        else:
            d = [((px[i, :3] - table[j]) ** 2).sum() for j in usable]
            sel = usable[int(np.argmin(d))]
        idx |= sel << (2 * i)
    return struct.pack("<HHI", c0, c1, idx)


def bc4_decode(block):
    v0, v1 = block[0], block[1]
    bits = int.from_bytes(block[2:8], "little")
    if v0 > v1:
        table = [v0, v1] + [((7 - k) * v0 + k * v1) // 7 for k in range(1, 7)]
    else:
        table = [v0, v1] + [((5 - k) * v0 + k * v1) // 5 for k in range(1, 5)] + [0, 255]
    out = np.array([table[(bits >> (3 * i)) & 7] for i in range(16)], dtype=np.uint8)
    return out.reshape(4, 4)


def bc4_encode(px):
    px = px.reshape(16).astype(np.int32)
    lo, hi = int(px.min()), int(px.max())
    if lo == hi:
        return bytes([hi, lo]) + b"\x00" * 6
    v0, v1 = hi, lo  # v0 > v1 -> 8 interpolated levels
    table = [v0, v1] + [((7 - k) * v0 + k * v1) // 7 for k in range(1, 7)]
    bits = 0
    for i in range(16):
        sel = int(np.argmin([abs(px[i] - t) for t in table]))
        bits |= sel << (3 * i)
    return bytes([v0, v1]) + bits.to_bytes(6, "little")


# ---------------------------------------------------------------- command streams

def _read_blocks(data, pos, rect, block_size, decode, prev, copy_prev, pix_shape):
    """prev is (pixels, rect) of the same layer in the previous frame; copied blocks
    are matched by canvas position, since the layer box can move between frames."""
    x1, y1, x2, y2 = rect
    wb, hb = (x2 - x1) // 4, (y2 - y1) // 4
    ncmd, = struct.unpack_from("<H", data, pos); pos += 2
    cmds = [struct.unpack_from("<BB", data, pos + 2 * i) for i in range(ncmd)]
    pos += 2 * ncmd
    out = np.zeros((hb * 4, wb * 4) + pix_shape, dtype=np.uint8)
    b = 0
    for skip, draw in cmds:
        for _ in range(skip):
            if copy_prev and prev is not None:
                by, bx = divmod(b, wb)
                ppix, (px1, py1, _, _) = prev
                sy, sx = y1 + by * 4 - py1, x1 + bx * 4 - px1
                if 0 <= sy <= ppix.shape[0] - 4 and 0 <= sx <= ppix.shape[1] - 4:
                    out[by*4:by*4+4, bx*4:bx*4+4] = ppix[sy:sy+4, sx:sx+4]
            b += 1
        for _ in range(draw):
            by, bx = divmod(b, wb)
            out[by*4:by*4+4, bx*4:bx*4+4] = decode(data[pos:pos + block_size])
            pos += block_size
            b += 1
    return out, pos


def _write_blocks(img, empty, encode):
    """img dims are multiples of 4. empty(block)->bool. Returns command bytes + block bytes."""
    return _write_blocks_with(img, lambda by, bx: empty(img[by*4:by*4+4, bx*4:bx*4+4]), encode)


# ---------------------------------------------------------------- file level

def read(path):
    data = open(path, "rb").read()
    sig, ver, nframes, u1, u2, u3 = struct.unpack_from("<4sHHHHI", data, 0)
    assert sig == b"SLDX", sig
    header = dict(version=ver, unk1=u1, unk2=0x10, unk3=u3 & 0xFFFF)
    pos = u2   # header size: 16 in most files, 14 in some (e.g. hunting chickens, guanaco)
    frames = []
    prev_layers = {}
    for _ in range(nframes):
        cw, ch, hx, hy, ftype, funk, fidx = struct.unpack_from("<HHhhBBH", data, pos)
        pos += 12
        frame = dict(width=cw, height=ch, hotspot=(hx, hy), type=ftype, unk=funk, index=fidx, layers={})
        main_rect = None
        for lt in LAYER_ORDER:
            if not ftype & lt:
                continue
            start = pos
            length, = struct.unpack_from("<I", data, pos)
            body = pos + 4
            layer = dict(raw=data[start:start + length])
            if lt in (LAYER_MAIN, LAYER_SHADOW):
                x1, y1, x2, y2, flag, unk = struct.unpack_from("<HHHHBB", data, body)
                layer.update(rect=(x1, y1, x2, y2), flag=flag, unk=unk)
                if lt == LAYER_MAIN:
                    main_rect = (x1, y1, x2, y2)
                p = body + 10
            elif lt in (LAYER_DAMAGE, LAYER_PC):
                flag, unk = struct.unpack_from("<BB", data, body)
                layer.update(flag=flag, unk=unk, rect=main_rect)
                # a few frames carry a mask but no main image; there's nothing to size it by
                p = body + 2 if main_rect is not None else None
            else:
                p = None
            if p is not None:
                rgba = lt in (LAYER_MAIN, LAYER_DAMAGE)
                pix, _ = _read_blocks(
                    data, p, layer["rect"], 8, bc1_decode if rgba else bc4_decode,
                    prev_layers.get(lt), bool(layer["flag"] & FLAG_COPY_PREV),
                    (4,) if rgba else ())
                layer["pixels"] = pix
                prev_layers[lt] = (pix, layer["rect"])
            frame["layers"][LAYER_NAMES[lt]] = layer
            pos = start + length + ((4 - length) % 4)
        frames.append(frame)
    return header, frames


def write(path, header, frames):
    """frames: list of dicts with width, height, hotspot, unk, and layers:
         main: dict(rect=(x1,y1,x2,y2), pixels=RGBA)  (rect width/height multiple of 4)
         shadow: dict(rect=..., pixels=HxW)  optional
         playercolor: dict(pixels=HxW, same size as main) optional
         damage: dict(pixels=RGBA, same size as main) optional
    """
    out = [struct.pack("<4sHHHHI", b"SLDX", header.get("version", 4), len(frames),
                       header.get("unk1", 0), header.get("unk2", 0x10), header.get("unk3", 0xFF))]
    for i, fr in enumerate(frames):
        ftype = 0
        layer_bytes = []
        L = fr["layers"]
        if "main" in L and "unknown" not in L:
            # every game sprite carries a block silhouette (used for hidden-unit outlines);
            # build one from the image instead of leaving it out
            L = dict(L, unknown=dict(raw=outline.encode(L["main"]["pixels"][..., 3] > 127, margin=1)))
        for lt in LAYER_ORDER:
            name = LAYER_NAMES[lt]
            if name not in L:
                continue
            ly = L[name]
            if lt == LAYER_UNK:
                body = ly["raw"][4:]  # silhouette: passed through, or generated below
            elif lt == LAYER_MAIN:
                x1, y1, x2, y2 = ly["rect"]
                body = struct.pack("<HHHHBB", x1, y1, x2, y2, BASE_FLAG[lt], ly.get("unk", 1))
                body += _write_blocks(ly["pixels"], lambda b: (b[..., 3] < 128).all(), bc1_encode)
            elif lt == LAYER_SHADOW:
                x1, y1, x2, y2 = ly["rect"]
                body = struct.pack("<HHHHBB", x1, y1, x2, y2, BASE_FLAG[lt], ly.get("unk", 1))
                body += _write_blocks(ly["pixels"], lambda b: (b == 0).all(), bc4_encode)
            elif lt == LAYER_DAMAGE:
                body = struct.pack("<BB", BASE_FLAG[lt], ly.get("unk", 1))
                body += _write_blocks_masked(ly["pixels"], L["main"]["pixels"], bc1_encode)
            elif lt == LAYER_PC:
                body = struct.pack("<BB", BASE_FLAG[lt], ly.get("unk", 1))
                body += _write_blocks_masked(ly["pixels"], L["main"]["pixels"], bc4_encode)
            length = 4 + len(body)
            chunk = struct.pack("<I", length) + body
            chunk += b"\x00" * ((4 - length) % 4)
            layer_bytes.append(chunk)
            ftype |= lt
        hx, hy = fr["hotspot"]
        out.append(struct.pack("<HHhhBBH", fr["width"], fr["height"], hx, hy, ftype, fr.get("unk", 0), i))
        out.extend(layer_bytes)
    open(path, "wb").write(b"".join(out))


def _write_blocks_masked(img, main, encode):
    # mask layers draw exactly the blocks the main layer draws
    return _write_blocks_with(img, lambda by, bx: (main[by*4:by*4+4, bx*4:bx*4+4, 3] < 128).all(), encode)


def _write_blocks_with(img, empty_at, encode):
    h, w = img.shape[:2]
    wb, hb = w // 4, h // 4
    flags, picked = [], []
    for by in range(hb):
        for bx in range(wb):
            if empty_at(by, bx):
                flags.append(False)
            else:
                flags.append(True)
                picked.append(img[by*4:by*4+4, bx*4:bx*4+4])
    cmds, i, n = [], 0, len(flags)
    while i < n:
        skip = 0
        while i < n and not flags[i] and skip < 255:
            skip += 1; i += 1
        draw = 0
        while i < n and flags[i] and draw < 255:
            draw += 1; i += 1
        cmds.append((skip, draw))
    many = {bc1_encode: bc1_encode_many, bc4_encode: bc4_encode_many}[encode]
    blocks = many(np.array(picked)) if picked else b""
    return (struct.pack("<H", len(cmds)) + b"".join(struct.pack("<BB", s, d) for s, d in cmds)
            + blocks)


# ---------------------------------------------------------------- vectorised encoders

def _rgb565_many(rgb):
    rgb = np.clip(np.round(rgb), 0, 255).astype(np.int64)
    return ((rgb[..., 0] >> 3) << 11) | ((rgb[..., 1] >> 2) << 5) | (rgb[..., 2] >> 3)


def _565_rgb_many(c):
    r = (c >> 11) & 31; g = (c >> 5) & 63; b = c & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.float64)


def bc1_encode_many(blocks):
    """blocks: N x 4 x 4 x 4 RGBA -> N*8 bytes, same rules as bc1_encode."""
    N = len(blocks)
    px = blocks.reshape(N, 16, 4).astype(np.float64)
    opaque = px[..., 3] >= 128
    cnt = np.maximum(opaque.sum(1), 1)
    rgb = px[..., :3]
    mean = (rgb * opaque[..., None]).sum(1) / cnt[:, None]
    cen = (rgb - mean[:, None]) * opaque[..., None]
    cov = np.einsum("nki,nkj->nij", cen, cen)
    axis = np.ones((N, 3)) / np.sqrt(3)
    for _ in range(8):  # power iteration for the principal colour axis
        axis = np.einsum("nij,nj->ni", cov, axis)
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    proj = np.einsum("nki,ni->nk", cen, axis)
    pmax = np.where(opaque, proj, -1e9).max(1)
    pmin = np.where(opaque, proj, 1e9).min(1)
    pmax = np.where(opaque.any(1), pmax, 0); pmin = np.where(opaque.any(1), pmin, 0)
    c0 = _rgb565_many(mean + axis * pmax[:, None])
    c1 = _rgb565_many(mean + axis * pmin[:, None])
    transparent = ~opaque.all(1)
    # transparent blocks need c0 <= c1, solid ones c0 > c1 (when they differ)
    swap = np.where(transparent, c0 > c1, c0 < c1)
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    p0, p1 = _565_rgb_many(c0), _565_rgb_many(c1)
    four = c0 > c1
    pal = np.stack([p0, p1,
                    np.where(four[:, None], (2 * p0 + p1) / 3, (p0 + p1) / 2),
                    np.where(four[:, None], (p0 + 2 * p1) / 3, 1e9)], 1)  # N x 4 x 3
    d = ((rgb[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)  # N x 16 x 4
    sel = d.argmin(-1)
    sel = np.where(opaque, sel, 3)
    idx = (sel.astype(np.uint64) << (2 * np.arange(16, dtype=np.uint64))).sum(1)
    out = np.zeros(N, dtype=[("c0", "<u2"), ("c1", "<u2"), ("i", "<u4")])
    out["c0"] = c0; out["c1"] = c1; out["i"] = idx
    return out.tobytes()


def bc4_encode_many(blocks):
    N = len(blocks)
    px = blocks.reshape(N, 16).astype(np.int64)
    hi, lo = px.max(1), px.min(1)
    k = np.arange(1, 7)
    pal = np.concatenate([hi[:, None], lo[:, None],
                          ((7 - k)[None] * hi[:, None] + k[None] * lo[:, None]) // 7], 1)  # N x 8
    sel = np.abs(px[:, :, None] - pal[:, None, :]).argmin(-1)
    sel = np.where((hi == lo)[:, None], 0, sel)
    bits = (sel.astype(np.uint64) << (3 * np.arange(16, dtype=np.uint64))).sum(1)
    out = np.zeros((N, 8), np.uint8)
    out[:, 0] = hi; out[:, 1] = lo
    for b in range(6):
        out[:, 2 + b] = (bits >> np.uint64(8 * b)) & np.uint64(255)
    return out.tobytes()
