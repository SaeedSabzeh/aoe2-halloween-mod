"""The per-frame silhouette layer of .sld files (layer bit 0x04).

One 16-bit mask per 4x4 block of the main layer (bit i = pixel i, row-major).
Per block row: a leading byte (omitted when the shape starts at column 0) skips empty
blocks before the shape; after that,
bytes with the top bit set (0x80 | n) are followed by n masks, and bytes
without it skip n fully solid blocks. Trailing empty blocks are not stored.
A table of row start offsets (rows 1..) precedes the data.
"""
import struct
import numpy as np


def _masks(alpha):
    h, w = alpha.shape
    b = alpha.reshape(h // 4, 4, w // 4, 4).swapaxes(1, 2).reshape(h // 4, w // 4, 16)
    return (b.astype(np.int64) << np.arange(16)).sum(-1)


def encode(alpha, margin=0):
    """alpha: HxW bool (H, W multiples of 4) -> raw layer bytes. margin grows the shape by that
    many pixels, like the slightly generous silhouettes in the game's own files."""
    for _ in range(margin):
        alpha = alpha | np.roll(alpha, 1, 0) | np.roll(alpha, -1, 0) | np.roll(alpha, 1, 1) | np.roll(alpha, -1, 1)
    m = _masks(alpha)
    hb, wb = m.shape
    data, offs = bytearray(), []
    for r in range(hb):
        if r > 0:
            offs.append(len(data))
        row = m[r]
        nz = np.nonzero(row)[0]
        if len(nz) == 0:
            n = wb
            while n > 0:
                data.append(min(n, 127)); n -= min(n, 127)
            continue
        s, e = nz[0], nz[-1]
        if s > 0:   # the game omits the leading skip when the shape starts at column 0
            data.append(int(s))
        c = s
        # solid blocks at the very end of a row must be spelled out: a final skip means empty
        tail = e
        while tail >= s and row[tail] == 0xFFFF:
            tail -= 1
        while c <= e:
            if row[c] == 0xFFFF and c <= tail:
                n = 0
                while c <= tail and row[c] == 0xFFFF and n < 127:
                    n += 1; c += 1
                data.append(n)
            else:
                run = []
                while c <= e and (row[c] != 0xFFFF or c > tail) and len(run) < 127:
                    run.append(int(row[c])); c += 1
                data.append(0x80 | len(run))
                for v in run:
                    data += struct.pack("<H", v)
        if e + 1 < wb:   # like the game, every row adds up to the full width
            data.append(int(wb - e - 1))
    body = bytes([5, 0, 0, 0]) + struct.pack("<%dH" % len(offs), *offs) + bytes(data)
    return struct.pack("<I", 4 + len(body)) + body


def decode(raw, wb, hb):
    """raw layer bytes -> HxW bool silhouette (inverse of encode)."""
    n_off = hb - 1
    offs = struct.unpack_from("<%dH" % n_off, raw, 8) if n_off > 0 else ()
    length = struct.unpack_from("<I", raw, 0)[0]
    data = raw[8 + 2 * n_off:length]
    m = np.zeros((hb, wb), np.int64)
    for r in range(hb):
        p = 0 if r == 0 else offs[r - 1]
        end = offs[r] if r < n_off else len(data)
        c = 0
        if p < end and not data[p] & 0x80:   # leading run of empty blocks
            c = data[p]; p += 1
        while p < end and c < wb:
            b = data[p]; p += 1
            if b & 0x80:
                for _ in range(b & 0x7F):
                    if c < wb:
                        m[r, c] = data[p] | (data[p + 1] << 8)
                    p += 2; c += 1
            else:
                if p < end:   # a skip that ends the row is trailing empty space, not solid
                    m[r, c:c + b] = 0xFFFF
                c += b
    bits = ((m[..., None] >> np.arange(16)) & 1).astype(bool)
    return bits.reshape(hb, wb, 4, 4).swapaxes(1, 2).reshape(hb * 4, wb * 4)
