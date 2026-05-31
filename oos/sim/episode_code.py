"""Reversible 'episode code' — a compact token that round-trips a full episode
spec (facility + level knobs + reset seed) so an exact layout can be regenerated.

This is NOT a hash (a hash is one-way). It *encodes* the parameters so they can
be decoded and the identical initial state rebuilt: `(level knobs + seed)`
deterministically reproduce the `InitialStateSampler` layout, the target
selection, AND the Poisson store-stream / dwell draws — every source of
randomness in `Environment.reset` is derived from `default_rng(seed)`. Full
`float64` precision is preserved, which is the whole point: the rounded `lvl`
log line (`bsf0.39…`) cannot reproduce a layout, but this can.

Format:  ``OOS1-<urlsafe_base64(struct payload)>`` — a fixed little-endian pack
(~70 bytes → ~95-char code). Categorical knobs are small stable index tables
(append-only, so old codes keep decoding); the facility is a length-prefixed
string so it stays robust to facility-set changes.
"""

from __future__ import annotations

import base64
import struct

_MAGIC = "OOS1"
# v2: rooms stopped being storage slots (the primitive-actions refactor). The
# room_state knob now stages a carrier docked at the room holding the pallet,
# which changes the sampler's RNG draw order — so v1 codes decode to a
# different layout under the new sampler. Bumped to invalidate them cleanly.
_VERSION = 2

# Stable index tables — APPEND ONLY (never reorder/remove; old codes hold indices).
_TASK = ("retrieve", "bring_empty")
_FROM = ("big", "small")
_ROUTE = ("direct", "handoff")
_ROOM = ("empty", "small_item", "big_item")

# The float knobs, in fixed pack order.
_FLOATS = (
    "big_shelf_fullness", "system_fullness", "big_ratio",
    "big_disorder", "small_disorder",
)


def _idx(table: tuple, value: str, field: str) -> int:
    try:
        return table.index(str(value))
    except ValueError:
        raise ValueError(f"episode_code: {field}={value!r} not in {table}") from None


def encode_episode(facility: str, params: dict, seed: int) -> str:
    """Pack `(facility, level knobs, seed)` into a copy-paste-safe code."""
    fb = str(facility).encode("utf-8")
    if len(fb) > 255:
        raise ValueError("episode_code: facility name too long")
    raw = struct.pack("<BqB", _VERSION, int(seed), len(fb)) + fb
    raw += struct.pack(
        "<BBBB",
        _idx(_TASK, params["task"], "task"),
        _idx(_FROM, params["retrieve_from"], "retrieve_from"),
        _idx(_ROUTE, params["retrieve_route"], "retrieve_route"),
        _idx(_ROOM, params["room_state"], "room_state"),
    )
    raw += struct.pack(
        "<BBH",
        int(params["target_depth"]) & 0xFF,
        1 if params.get("require_solvable", True) else 0,
        int(params.get("max_solvable_retries", 50)) & 0xFFFF,
    )
    raw += struct.pack("<5d", *(float(params[k]) for k in _FLOATS))
    b64 = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return f"{_MAGIC}-{b64}"


def decode_episode(code: str) -> dict:
    """Inverse of `encode_episode`. Returns
    ``{"facility": str, "seed": int, "params": dict}``. Raises ValueError on a
    malformed / wrong-version code."""
    code = code.strip()
    if code.startswith(_MAGIC + "-"):
        body = code[len(_MAGIC) + 1:]
    elif "-" not in code:
        body = code   # tolerate a bare base64 body (magic stripped on copy)
    else:
        raise ValueError(f"not an episode code (expected {_MAGIC}- prefix): {code[:16]}…")
    try:
        raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        ver, seed, flen = struct.unpack_from("<BqB", raw, 0)
        if ver != _VERSION:
            raise ValueError(f"unsupported episode-code version {ver}")
        off = struct.calcsize("<BqB")
        facility = raw[off:off + flen].decode("utf-8")
        off += flen
        ti, fi, ri, mi = struct.unpack_from("<BBBB", raw, off)
        off += 4
        depth, solv, retries = struct.unpack_from("<BBH", raw, off)
        off += struct.calcsize("<BBH")
        floats = struct.unpack_from("<5d", raw, off)
        params = {
            "task": _TASK[ti],
            "retrieve_from": _FROM[fi],
            "retrieve_route": _ROUTE[ri],
            "room_state": _ROOM[mi],
            "target_depth": int(depth),
            "require_solvable": bool(solv),
            "max_solvable_retries": int(retries),
        }
        params.update({k: float(v) for k, v in zip(_FLOATS, floats)})
        return {"facility": facility, "seed": int(seed), "params": params}
    except (ValueError, KeyError, IndexError, struct.error) as e:
        raise ValueError(f"corrupt episode code: {type(e).__name__}: {e}") from e
