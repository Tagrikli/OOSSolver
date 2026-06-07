"""Reversible 'layout code' — a compact token that round-trips a generated
layout so an EXACT state can be reproduced or shared.

A viz layout is fully determined by `shuffle_state(engine, fullness,
rng=default_rng(seed))` on a given facility, so encoding `(facility, fullness,
seed)` is enough to rebuild the identical state. Full float64 `fullness` is
preserved (a rounded display value can't reproduce a layout; this can).

Format: ``OOS2-<urlsafe_base64(struct payload)>`` — a fixed little-endian pack
(seed:int64, fullness:float64, then a length-prefixed facility string).
"""

from __future__ import annotations

import base64
import struct

_MAGIC = "OOS2"
_HEAD = "<qdB"   # seed (int64), fullness (float64), facility-name length (uint8)


def encode_layout(facility: str, fullness: float, seed: int) -> str:
    """Pack `(facility, fullness, seed)` into a copy-paste-safe code."""
    fb = str(facility).encode("utf-8")
    if len(fb) > 255:
        raise ValueError("layout_code: facility name too long")
    raw = struct.pack(_HEAD, int(seed), float(fullness), len(fb)) + fb
    b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{_MAGIC}-{b64}"


def decode_layout(code: str) -> dict:
    """Inverse of `encode_layout`. Returns
    ``{"facility": str, "fullness": float, "seed": int}``. Raises ValueError on
    a malformed / wrong-version code."""
    code = code.strip()
    if code.startswith(_MAGIC + "-"):
        body = code[len(_MAGIC) + 1:]
    elif "-" not in code:
        body = code   # tolerate a bare base64 body (magic stripped on copy)
    else:
        raise ValueError(f"not a layout code (expected {_MAGIC}- prefix): {code[:16]}…")
    try:
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        seed, fullness, flen = struct.unpack_from(_HEAD, raw, 0)
        off = struct.calcsize(_HEAD)
        facility = raw[off:off + flen].decode("utf-8")
        return {"facility": facility, "fullness": float(fullness), "seed": int(seed)}
    except (ValueError, IndexError, struct.error) as e:
        raise ValueError(f"corrupt layout code: {type(e).__name__}: {e}") from e
