"""Manual sim editing handlers — pallet hover-edits and shelf push/pop.

The mouse-over-a-pallet shortcuts (1/2/3 to set contents, 4/5 to pop/push
on the hovered shelf) and the queue-button clicks (queue small/big/clear,
randomize) all live here so app.py's event loop reads as dispatch instead
of inline edit code.

Each handler takes the already-resolved targets (a pallet id, a shelf id,
a button label) plus the player/facility/toasts and applies the mutation
+ refreshes the player's cached obs/info.
"""

from __future__ import annotations

import numpy as np

from oos.sim.state import Pallet
from oos.viz.components import ToastManager
from oos.viz.player import Player


def _refresh_player(player: Player, facility) -> None:
    """Re-pull obs/info so the next policy call sees the new live state.

    Mutating shelves/queue out-of-band makes the env's cached decoder
    stale; this rebuild keeps the agent's view byte-identical to the
    human's. Used after every manual edit."""
    player.env.refresh_decision_context()  # type: ignore[attr-defined]
    player.obs, player.info = (
        player.env._observation_for_current(  # type: ignore[attr-defined]
            facility, dt=0.0, completions=[], arrivals=[],
        )
    )


# ---------------------------------------------------------------------------
# Queue panel buttons (manual mode)
# ---------------------------------------------------------------------------


def handle_queue_button(
    btn: str,
    facility,
    player: Player,
    fullness: float,
    toasts: ToastManager,
) -> bool:
    """Apply the action for a queue-panel button click. Returns True iff
    the click was a known button (so the caller can short-circuit)."""
    if btn == "queue small":
        facility.enqueue_store("small")
        _refresh_player(player, facility)
        toasts.accent("+ STORE small", lifetime=2.0)
        return True
    if btn == "queue big":
        facility.enqueue_store("big")
        _refresh_player(player, facility)
        toasts.accent("+ STORE big", lifetime=2.0)
        return True
    if btn == "queue clear":
        facility.clear_queue()
        _refresh_player(player, facility)
        toasts.warn("QUEUE CLEARED", lifetime=2.0)
        return True
    if btn == "randomize":
        from oos.sim.shuffle import shuffle_state
        shuffle_state(
            facility,
            fullness=fullness,
            rng=np.random.default_rng(),
            require_solvable=True,
        )
        facility.clear_queue()
        _refresh_player(player, facility)
        toasts.accent("STATE RANDOMIZED", lifetime=2.5)
        return True
    return False


def handle_pallet_click(
    pallet_id: int,
    facility,
    player: Player,
    toasts: ToastManager,
) -> None:
    """Left-click on a pallet toggles a Retrieve for it."""
    now_pending = facility.toggle_retrieve_for_pallet(pallet_id)
    _refresh_player(player, facility)
    if now_pending:
        toasts.info(f"+ RETRIEVE pallet={pallet_id}", lifetime=2.5)
    else:
        toasts.warn(f"– RETRIEVE pallet={pallet_id}", lifetime=2.5)


# ---------------------------------------------------------------------------
# Pallet-hover number keys (1/2/3 set contents)
# ---------------------------------------------------------------------------


def set_pallet_contents(
    pallet_id: int,
    target_contents: str,
    facility,
    player: Player,
    toasts: ToastManager,
) -> None:
    """Replace the pallet's contents in-place; reject big-on-small."""
    owner_sid: str | None = None
    owner_idx = -1
    for sid, ss in facility.state.shelves.items():
        for i, p in enumerate(ss.stack):
            if p.id == pallet_id:
                owner_sid = sid
                owner_idx = i
                break
        if owner_sid is not None:
            break
    if owner_sid is None:
        toasts.warn(f"pallet {pallet_id} not on a shelf", lifetime=2.0)
        return
    shelf_topo = facility.topology.shelves[owner_sid]
    if target_contents == "big" and shelf_topo.size_class != "big":
        toasts.warn("BIG item not allowed on small shelf", lifetime=2.0)
        return
    old = facility.state.shelves[owner_sid].stack[owner_idx]
    facility.state.shelves[owner_sid].stack[owner_idx] = (
        Pallet(id=old.id, contents=target_contents)  # type: ignore[arg-type]
    )
    _refresh_player(player, facility)
    toasts.accent(f"pallet {pallet_id} → {target_contents}", lifetime=2.0)


# ---------------------------------------------------------------------------
# Shelf-hover keys (4 = pop top, 5 = push empty)
# ---------------------------------------------------------------------------


def pop_shelf_top(
    shelf_id: str,
    facility,
    player: Player,
    toasts: ToastManager,
) -> None:
    ss = facility.state.shelves[shelf_id]
    if not ss.stack:
        toasts.warn(f"shelf {shelf_id} already empty", lifetime=2.0)
        return
    popped = ss.stack.pop()
    toasts.accent(f"removed pallet {popped.id} from {shelf_id}", lifetime=2.0)
    _refresh_player(player, facility)


def push_empty_pallet(
    shelf_id: str,
    facility,
    player: Player,
    toasts: ToastManager,
) -> None:
    ss = facility.state.shelves[shelf_id]
    shelf_topo = facility.topology.shelves[shelf_id]
    if len(ss.stack) >= shelf_topo.capacity:
        toasts.warn(
            f"shelf {shelf_id} at capacity ({shelf_topo.capacity})",
            lifetime=2.0,
        )
        return
    new_pid = facility._next_pallet_id
    facility._next_pallet_id += 1
    ss.stack.append(Pallet(id=new_pid, contents="empty"))
    toasts.accent(f"pushed empty pallet {new_pid} → {shelf_id}", lifetime=2.0)
    _refresh_player(player, facility)
