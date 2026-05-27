"""Manual sim editing handlers — pallet hover-edits and shelf push/pop.

The mouse-over-a-pallet shortcuts (1/2/3 to set contents, 4/5 to pop/push
on the hovered shelf) and the queue-button clicks (queue small/big/clear,
randomize) all live here so app.py's event loop reads as dispatch instead
of inline edit code.

Each handler takes the already-resolved targets (a pallet id, a shelf id,
a button label) plus the agent/facility/toasts and applies the mutation
+ refreshes the agent's cached obs/info via the env's
`refresh_decision_context`.
"""

from __future__ import annotations

import numpy as np

from oos.agent import Agent
from oos.facility import Facility
from oos.sim.state import Pallet
from oos.viz.components import ToastManager


def _refresh_agent(agent: Agent) -> None:
    """Re-pull obs/info so the next policy call sees the new live state.

    Mutating shelves/queue out-of-band makes the env's cached decoder
    stale; this rebuild keeps the agent's view byte-identical to the
    human's. Used after every manual edit.
    """
    env = agent.facility.env
    env.refresh_decision_context()
    agent.obs, agent.info = env._observation_for_current(  # type: ignore[attr-defined]
        agent.facility.sim, dt=0.0, completions=[], arrivals=[],
    )


# ---------------------------------------------------------------------------
# Queue panel buttons (manual mode)
# ---------------------------------------------------------------------------


def handle_queue_button(
    btn: str,
    facility: Facility,
    agent: Agent,
    fullness: float,
    toasts: ToastManager,
) -> bool:
    """Apply the action for a queue-panel button click. Returns True iff
    the click was a known button (so the caller can short-circuit)."""
    sim = facility.sim
    if btn == "queue small":
        sim.enqueue_store("small")
        _refresh_agent(agent)
        toasts.accent("+ STORE small", lifetime=2.0)
        return True
    if btn == "queue big":
        sim.enqueue_store("big")
        _refresh_agent(agent)
        toasts.accent("+ STORE big", lifetime=2.0)
        return True
    if btn == "queue clear":
        sim.clear_queue()
        _refresh_agent(agent)
        toasts.warn("QUEUE CLEARED", lifetime=2.0)
        return True
    if btn == "randomize":
        from oos.sim.shuffle import shuffle_state
        shuffle_state(
            sim,
            fullness=fullness,
            rng=np.random.default_rng(),
            require_solvable=True,
        )
        sim.clear_queue()
        _refresh_agent(agent)
        toasts.accent("STATE RANDOMIZED", lifetime=2.5)
        return True
    return False


def handle_pallet_click(
    pallet_id: int,
    facility: Facility,
    agent: Agent,
    toasts: ToastManager,
) -> None:
    """Left-click on a pallet toggles a Retrieve for it."""
    now_pending = facility.sim.toggle_retrieve_for_pallet(pallet_id)
    _refresh_agent(agent)
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
    facility: Facility,
    agent: Agent,
    toasts: ToastManager,
) -> None:
    """Replace the pallet's contents in-place; reject big-on-small."""
    sim = facility.sim
    owner_sid: str | None = None
    owner_idx = -1
    for sid, ss in sim.state.shelves.items():
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
    shelf_topo = sim.topology.shelves[owner_sid]
    if target_contents == "big" and shelf_topo.size_class != "big":
        toasts.warn("BIG item not allowed on small shelf", lifetime=2.0)
        return
    old = sim.state.shelves[owner_sid].stack[owner_idx]
    sim.state.shelves[owner_sid].stack[owner_idx] = (
        Pallet(id=old.id, contents=target_contents)  # type: ignore[arg-type]
    )
    _refresh_agent(agent)
    toasts.accent(f"pallet {pallet_id} → {target_contents}", lifetime=2.0)


# ---------------------------------------------------------------------------
# Shelf-hover keys (4 = pop top, 5 = push empty)
# ---------------------------------------------------------------------------


def pop_shelf_top(
    shelf_id: str,
    facility: Facility,
    agent: Agent,
    toasts: ToastManager,
) -> None:
    ss = facility.sim.state.shelves[shelf_id]
    if not ss.stack:
        toasts.warn(f"shelf {shelf_id} already empty", lifetime=2.0)
        return
    popped = ss.stack.pop()
    toasts.accent(f"removed pallet {popped.id} from {shelf_id}", lifetime=2.0)
    _refresh_agent(agent)


def push_empty_pallet(
    shelf_id: str,
    facility: Facility,
    agent: Agent,
    toasts: ToastManager,
) -> None:
    sim = facility.sim
    ss = sim.state.shelves[shelf_id]
    shelf_topo = sim.topology.shelves[shelf_id]
    if len(ss.stack) >= shelf_topo.capacity:
        toasts.warn(
            f"shelf {shelf_id} at capacity ({shelf_topo.capacity})",
            lifetime=2.0,
        )
        return
    new_pid = sim._next_pallet_id  # type: ignore[attr-defined]
    sim._next_pallet_id += 1       # type: ignore[attr-defined]
    ss.stack.append(Pallet(id=new_pid, contents="empty"))
    toasts.accent(f"pushed empty pallet {new_pid} → {shelf_id}", lifetime=2.0)
    _refresh_agent(agent)
