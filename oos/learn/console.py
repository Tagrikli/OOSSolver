"""Console logging for trainers — the styled lines every run prints.

What a trainer prints doesn't vary in shape: a config banner, then per
iteration an iter header + an episode line + a PPO line, plus the occasional
eval line. This wraps the cyberpunk palette in `oos.learn._style` into those
fixed log calls so trainers don't re-hand-format ANSI strings (the way they all
used to). The style primitives are re-exported for the few bespoke `kv` values a
trainer's banner needs.

    console.banner("retrieve · tiny")
    console.kv("reward", f"{console.v('DELIVER')} ...")
    console.log_iter(it, total_env_steps, wall, collect_secs, update_secs)
    console.log_episode(mean_ret, mean_len, sampled_rate, n_eps)
    console.log_ppo(metrics)               # the PPOMetrics from ppo_update
    console.log_eval(rate, detail="@ depth 0, 50 layouts, argmax")
"""

from __future__ import annotations

from typing import Any

from oos.learn._style import (
    C_ANCHOR,
    C_DIM,
    C_RETURN,
    C_WALL,
    _C,
    _banner,
    _color_ev,
    _color_kl,
    _color_success,
    _kv,
    _v,
    _v_num,
)
from oos.learn.notify import notify  # re-exported: console.notify(title, body)

# Re-exports so trainers can compose custom banner values without importing _style.
v = _v
v_num = _v_num
DIM = C_DIM
RESET = _C.RESET


def banner(title: str) -> None:
    _banner(title)


def kv(label: str, value: str) -> None:
    _kv(label, value)


def fields(*pairs: "tuple[str, object]") -> str:
    """Compose one kv VALUE from `(dim-label, value)` pairs, rendered as
    `label value  label value`. An empty label prints the value alone — handy
    for a leading bare token (e.g. `("", "shuffle_state")`)."""
    out = []
    for label, value in pairs:
        out.append(f"{C_DIM}{label}{_C.RESET} {_v(value)}" if label else _v(value))
    return "  ".join(out)


def config_banner(title: str, rows: "list[tuple[str, str]]") -> None:
    """Print a run banner in one call: the title bar, then one key/value row per
    `(label, value)`. Build each value with `fields(...)` or a plain f-string."""
    banner(title)
    for label, value in rows:
        kv(label, value)


def log_iter(it: int, total_env_steps: int, wall: float,
             collect_secs: float, update_secs: float) -> None:
    print(
        f"{_C.BOLD}{_C.CYAN}━━━ iter {it:>4d} ━━━{_C.RESET}  "
        f"{C_DIM}env_steps{_C.RESET} {_v_num(f'{total_env_steps:>9,d}')}  "
        f"{C_DIM}wall{_C.RESET} {C_WALL}{wall:>5.0f}s{_C.RESET}  "
        f"{C_DIM}(collect {collect_secs:>4.1f}s + update {update_secs:>4.1f}s){_C.RESET}"
    )


def log_episode(mean_ret: float, mean_len: float,
                sampled_rate: float, n_eps: int) -> None:
    print(
        f"  {C_DIM}▎ episode{_C.RESET}   {C_DIM}return{_C.RESET} "
        f"{C_RETURN}{mean_ret:>+7.3f}{_C.RESET}   {C_DIM}ep_len{_C.RESET} "
        f"{_v(f'{mean_len:>6.1f}')}   {C_DIM}sampled{_C.RESET} "
        f"{_color_success(sampled_rate)}{sampled_rate * 100:>5.1f}%{_C.RESET}   "
        f"{C_DIM}n_eps{_C.RESET} {_v(f'{n_eps:>3d}')}"
    )


def log_ppo(metrics: Any) -> None:
    """Print the PPO line from a PPOMetrics-like object (policy_loss, value_loss,
    entropy, approx_kl, clip_fraction, explained_variance)."""
    print(
        f"  {C_DIM}▎ policy{_C.RESET}    {C_DIM}pi_loss{_C.RESET} "
        f"{_v(f'{metrics.policy_loss:>+7.3f}')}   {C_DIM}v_loss{_C.RESET} "
        f"{_v(f'{metrics.value_loss:>6.3f}')}   {C_DIM}entropy{_C.RESET} "
        f"{_v(f'{metrics.entropy:>5.3f}')}   {C_DIM}kl{_C.RESET} "
        f"{_color_kl(metrics.approx_kl)}{metrics.approx_kl:>+7.4f}{_C.RESET}   "
        f"{C_DIM}clip_frac{_C.RESET} {_v(f'{metrics.clip_fraction:>4.2f}')}   "
        f"{C_DIM}expl_var{_C.RESET} "
        f"{_color_ev(metrics.explained_variance)}{metrics.explained_variance:>+5.2f}{_C.RESET}"
    )


def log_eval(rate: float, *, label: str = "GREEDY", detail: str = "") -> None:
    print(
        f"  {_C.BOLD}{C_ANCHOR}▓ {label}{_C.RESET}   "
        f"{_color_success(rate)}{_C.BOLD}{rate * 100:>5.1f}%{_C.RESET}   "
        f"{C_DIM}{detail}{_C.RESET}"
    )
