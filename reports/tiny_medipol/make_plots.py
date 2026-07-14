"""Render the characterization graphs from reports/tiny_medipol/data/*.json.

    uv run --with matplotlib python reports/tiny_medipol/make_plots.py

Design: dataviz reference palette (light mode), sequential single-hue ramps
for fullness magnitude, fixed categorical order, thin marks, recessive
grid, no dual axes; prints a stats summary consumed by report.md.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PLOTS = os.path.join(HERE, "plots")
os.makedirs(PLOTS, exist_ok=True)

# Reference palette (light), roles per the dataviz method.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
BLUE = "#2a78d6"      # categorical slot 1
AQUA = "#1baf7a"      # slot 2
YELLOW = "#eda100"    # slot 3
VIOLET = "#4a3aa7"    # slot 5
RED = "#e34948"       # slot 6
GRAY = "#b9b8b2"      # de-emphasis


def seq_blues(n: int) -> list[str]:
    """Sequential ramp: one hue (slot-1 blue), light -> dark."""
    base = np.array([42, 120, 214]) / 255.0
    out = []
    for i in range(n):
        k = 0.75 - 0.75 * i / max(1, n - 1)      # 0.75 (light) -> 0 (full)
        c = base + (1.0 - base) * k
        out.append(matplotlib.colors.to_hex(c))
    return out


def style(ax, ylab="", xlab=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylab, color=INK2, fontsize=10)
    ax.set_xlabel(xlab, color=INK2, fontsize=10)


def fig_ax(w=7.0, h=4.0, ncols=1):
    fig, axs = plt.subplots(1, ncols, figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, axs


def load(name):
    with open(os.path.join(DATA, name + ".json")) as f:
        return json.load(f)


def save(fig, name, title):
    fig.suptitle(title, color=INK, fontsize=12, fontweight="bold", x=0.02,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(PLOTS, name), facecolor=SURFACE)
    plt.close(fig)
    print("plot:", name)


summary: dict = {}

# --- A: latency vs depth × fullness -----------------------------------------
d = load("exp_a_depth_fullness")
rows = [r for r in d["rows"] if r["latency"] is not None]
fulls = sorted({r["fullness"] for r in rows})
colors = seq_blues(len(fulls))
fig, ax = fig_ax()
rng = np.random.default_rng(0)
for f, c in zip(fulls, colors):
    xs, med = [], []
    for depth in (0, 1, 2):
        lats = [r["latency"] for r in rows
                if r["fullness"] == f and r["depth"] == depth]
        if not lats:
            continue
        for l in lats:      # faint raw points, jittered
            ax.plot(depth + rng.uniform(-0.12, 0.12), l, "o", ms=3.5,
                    color=c, alpha=0.25, mec="none")
        xs.append(depth)
        med.append(float(np.median(lats)))
    ax.plot(xs, med, "-o", color=c, lw=2, ms=6,
            label=f"fullness {f:.2f}", mec=SURFACE, mew=1)
style(ax, "delivery latency (s, incl. 45 s exit dwell)", "burial depth of the requested car")
ax.set_xticks([0, 1, 2])
ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
save(fig, "a_latency_depth_fullness.png",
     "Retrieval latency vs burial depth, by pool fullness (medians + raw runs)")
summary["a"] = {
    f"f{f}": {d_: round(float(np.median([r["latency"] for r in rows
              if r["fullness"] == f and r["depth"] == d_])), 1)
              for d_ in (0, 1, 2)
              if [r for r in rows if r["fullness"] == f and r["depth"] == d_]}
    for f in fulls}
summary["a_stuck"] = sum(1 for r in d["rows"] if r["stuck"])
summary["a_n"] = len(d["rows"])

# --- B: drain scaling ---------------------------------------------------------
d = load("exp_b_drain_scaling")
ks = sorted({r["k"] for r in d["rows"]})
fig, axs = fig_ax(9.5, 3.8, 2)
thr = [float(np.mean([r["throughput_per_h"] for r in d["rows"]
                      if r["k"] == k])) for k in ks]
thr_min = [float(np.min([r["throughput_per_h"] for r in d["rows"]
                         if r["k"] == k])) for k in ks]
thr_max = [float(np.max([r["throughput_per_h"] for r in d["rows"]
                         if r["k"] == k])) for k in ks]
axs[0].fill_between(ks, thr_min, thr_max, color=BLUE, alpha=0.15, lw=0)
axs[0].plot(ks, thr, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1)
style(axs[0], "drain throughput (cars/h)", "simultaneous requests (k)")
axs[0].set_xticks(ks)
p95 = [float(np.percentile(sum((r["latencies"] for r in d["rows"]
                                if r["k"] == k), []), 95)) for k in ks]
p50 = [float(np.percentile(sum((r["latencies"] for r in d["rows"]
                                if r["k"] == k), []), 50)) for k in ks]
axs[1].plot(ks, p50, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1,
            label="p50")
axs[1].plot(ks, p95, "-o", color=seq_blues(3)[0], lw=2, ms=6, mec=SURFACE,
            mew=1, label="p95")
style(axs[1], "per-car latency (s)", "simultaneous requests (k)")
axs[1].set_xticks(ks)
axs[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
save(fig, "b_drain_scaling.png",
     "Concurrent drain at fullness 0.70 — throughput and per-car latency vs k")
summary["b"] = {k: {"thr": round(t, 1), "p50": round(a, 0), "p95": round(b, 0)}
                for k, t, a, b in zip(ks, thr, p50, p95)}
summary["b_stuck"] = sum(1 for r in d["rows"] if r["stuck"])

# --- C: intake vs dwell -------------------------------------------------------
d = load("exp_c_intake_dwell")
dws = sorted({r["dwell"] for r in d["rows"]})
means = [float(np.mean([r["intake_per_h"] for r in d["rows"]
                        if r["dwell"] == dw])) for dw in dws]
fig, ax = fig_ax(5.6, 3.8)
bars = ax.bar([str(int(dw)) for dw in dws], means, width=0.55, color=BLUE,
              edgecolor=SURFACE, linewidth=2)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 1.2, f"{m:.0f}",
            ha="center", color=INK2, fontsize=9)
style(ax, "sustained intake (stores/h)", "customer dwell setting (s)")
save(fig, "c_intake_dwell.png",
     "Store intake rate vs customer dwell (20-sedan burst, 2 rooms)")
summary["c"] = dict(zip([int(x) for x in dws], [round(m, 1) for m in means]))

# --- D: SUV acceptance ---------------------------------------------------------
d = load("exp_d_suv_acceptance")
fulls = sorted({r["fullness"] for r in d["rows"]})
acc_big, acc_small = [], []
for f in fulls:
    rs = [r for r in d["rows"] if r["fullness"] == f]
    big_a = sum(r["big_arrivals"] for r in rs)
    big_r = sum(r["big_refused"] for r in rs)
    small_a = sum(r["small_arrivals"] for r in rs)
    small_r = sum(r["small_refused"] for r in rs)
    acc_big.append(100.0 * (1 - big_r / max(1, big_a)))
    acc_small.append(100.0 * (1 - small_r / max(1, small_a)))
fig, ax = fig_ax(6.4, 4.0)
ax.plot(fulls, acc_small, "-o", color=AQUA, lw=2, ms=6, mec=SURFACE, mew=1,
        label="sedan")
ax.plot(fulls, acc_big, "-o", color=BLUE, lw=2, ms=6, mec=SURFACE, mew=1,
        label="SUV")
for x, y in zip(fulls, acc_big):
    ax.text(x, y - 6, f"{y:.0f}%", ha="center", color=INK2, fontsize=8)
style(ax, "admission acceptance (%)", "pool fullness at start")
ax.set_ylim(0, 108)
ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
save(fig, "d_suv_acceptance.png",
     "Arrival admission acceptance vs fullness (2 h mixed stream, 35% SUV)")
summary["d"] = {f: round(a, 1) for f, a in zip(fulls, acc_big)}

# --- E: service ops -------------------------------------------------------------
d = load("exp_e_service_ops")
fulls = sorted({r["fullness"] for r in d["rows"]})
ev = [float(np.mean([r["evict_latency"] for r in d["rows"]
                     if r["fullness"] == f])) for f in fulls]
pl = [float(np.mean([r["place_latency"] for r in d["rows"]
                     if r["fullness"] == f and r["place_latency"]]))
      for f in fulls]
x = np.arange(len(fulls))
w = 0.32
fig, ax = fig_ax(6.4, 4.0)
ax.bar(x - w / 2, ev, w, color=BLUE, edgecolor=SURFACE, linewidth=2,
       label="Evict")
ax.bar(x + w / 2, pl, w, color=AQUA, edgecolor=SURFACE, linewidth=2,
       label="Place")
for xi, v in zip(x - w / 2, ev):
    ax.text(xi, v + 2, f"{v:.0f}", ha="center", color=INK2, fontsize=8)
for xi, v in zip(x + w / 2, pl):
    ax.text(xi, v + 2, f"{v:.0f}", ha="center", color=INK2, fontsize=8)
style(ax, "completion time (s)", "pool fullness")
ax.set_xticks(x, [f"{f:.2f}" for f in fulls])
ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
save(fig, "e_service_ops.png",
     "Charger service ops — mean completion time vs fullness")
ok_e = sum(1 for r in d["rows"] if r["evict_ok"])
ok_p = sum(1 for r in d["rows"] if r["place_ok"])
n_p = sum(1 for r in d["rows"] if r["place_ok"] is not None)
summary["e"] = {"evict_ok": f"{ok_e}/{len(d['rows'])}",
                "place_ok": f"{ok_p}/{n_p}",
                "evict_s": {f: round(v, 0) for f, v in zip(fulls, ev)},
                "place_s": {f: round(v, 0) for f, v in zip(fulls, pl)}}

# --- F: groom convergence --------------------------------------------------------
d = load("exp_f_groom")
fig, ax = fig_ax(6.6, 4.0)
for i, run in enumerate(d["runs"]):
    ts = [s["t"] / 60.0 for s in run["samples"]]
    nb = [s["nonbig_on_big"] for s in run["samples"]]
    ba = [s["big_air"] for s in run["samples"]]
    ax.plot(ts, nb, color=BLUE, lw=2 if i == 0 else 1.2,
            alpha=1.0 if i == 0 else 0.45,
            label="non-bigs on big shelves" if i == 0 else None)
    ax.plot(ts, ba, color=AQUA, lw=2 if i == 0 else 1.2,
            alpha=1.0 if i == 0 else 0.45,
            label="free big-shelf slots" if i == 0 else None)
style(ax, "count", "idle time (min)")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
save(fig, "f_groom_declutter.png",
     "Idle groom: big-shelf decluttering over time (3 seeds)")
last = d["runs"][0]["samples"][-1]
summary["f"] = {"final_nonbig": last["nonbig_on_big"],
                "final_big_air": last["big_air"],
                "moves": last["moves"]}

# --- G: prefetch A/B ---------------------------------------------------------------
d = load("exp_g_prefetch")
on = [r["restage_s"] for r in d["rows"] if r["prefetch"] and r["restage_s"]]
off = [r["restage_s"] for r in d["rows"]
       if not r["prefetch"] and r["restage_s"]]
fig, ax = fig_ax(5.4, 3.8)
vals = [float(np.mean(off)), float(np.mean(on))]
bars = ax.bar(["prefetch OFF", "prefetch ON"], vals, width=0.5,
              color=[GRAY, BLUE], edgecolor=SURFACE, linewidth=2)
for b, v, arr in zip(bars, vals, (off, on)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.8,
            f"{v:.0f} s  (n={len(arr)})", ha="center", color=INK2,
            fontsize=9)
style(ax, "store → room re-staged (s)", "")
save(fig, "g_prefetch_ab.png",
     "Staging prefetch A/B — re-stage time after a store (relay worlds)")
summary["g"] = {"on_s": round(vals[1], 1), "off_s": round(vals[0], 1)}

# --- month -------------------------------------------------------------------------
d = load("exp_month")
days = d["days"]
xs = [x["day"] for x in days]

fig, ax = fig_ax(8.2, 3.9)
ax.plot(xs, [x["deliveries"] for x in days], "-o", color=BLUE, lw=2, ms=4,
        mec=SURFACE, mew=0.8, label="deliveries")
ax.plot(xs, [x["stores"] for x in days], "-o", color=AQUA, lw=2, ms=4,
        mec=SURFACE, mew=0.8, label="stores")
ax.plot(xs, [x["dropped"] for x in days], "-o", color=YELLOW, lw=1.6, ms=4,
        mec=SURFACE, mew=0.8, label="SUVs refused")
style(ax, "per day", "day")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncols=3)
save(fig, "m1_throughput.png",
     "30-day endurance — daily task volume (tiny_medipol_ev, target 0.8)")

fig, ax = fig_ax(8.2, 3.9)
# A wedged day can deliver nothing (p50 = None) — mask it so the gap is
# visible in the drift plot instead of crashing the render.
p50 = np.array([np.nan if x["p50"] is None else x["p50"] for x in days])
p95 = np.array([np.nan if x["p95"] is None else x["p95"] for x in days])
mx = np.array([np.nan if x["max"] is None else x["max"] for x in days])
ax.fill_between(xs, p50, p95, color=BLUE, alpha=0.15, lw=0,
                label="p50–p95 band")
ax.plot(xs, p50, "-", color=BLUE, lw=2, label="p50")
ax.plot(xs, mx, "-", color=seq_blues(3)[0], lw=1.2, label="max")
style(ax, "delivery latency (s)", "day")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, ncols=3)
save(fig, "m2_latency_drift.png",
     "30-day endurance — daily delivery-latency percentiles (drift check)")

fig, axs = fig_ax(9.5, 3.8, 2)
axs[0].plot(xs, [x["replans"] for x in days], "-o", color=BLUE, lw=2, ms=4,
            mec=SURFACE, mew=0.8, label="replans")
axs[0].plot(xs, [x["moves"] / 10.0 for x in days], "-", color=GRAY, lw=1.4,
            label="moves ÷10")
style(axs[0], "per day", "day")
axs[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
axs[1].plot(xs, [100 * x["staged_uptime"] for x in days], "-o", color=AQUA,
            lw=2, ms=4, mec=SURFACE, mew=0.8)
style(axs[1], "staged-room uptime (%)", "day")
axs[1].set_ylim(0, 105)
save(fig, "m3_health.png",
     "30-day endurance — solver health (replans, motion, staging uptime)")

prof = d["profile_day1"]
if prof:
    fig, axs = fig_ax(9.5, 3.8, 2)
    t0 = prof[0]["t"]
    hrs = [(p["t"] - t0) / 3600.0 + 5.0 for p in prof]   # clock hours
    axs[0].plot(hrs, [p["pending_stores"] for p in prof], color=AQUA, lw=2,
                label="stores waiting")
    axs[0].plot(hrs, [p["pending_retrieves"] for p in prof], color=BLUE,
                lw=2, label="retrieves waiting")
    style(axs[0], "queue length", "clock hour")
    axs[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    axs[1].plot(hrs, [p["cars"] for p in prof], color=VIOLET, lw=2)
    style(axs[1], "cars in facility", "clock hour")
    save(fig, "m4_day_profile.png",
         "One representative day — queues and occupancy across the clock")

all_lats = sorted(sum((x["latencies"] for x in days), []))
fig, ax = fig_ax(6.8, 3.9)
ax.hist(all_lats, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
for q, name in ((50, "p50"), (95, "p95")):
    v = float(np.percentile(all_lats, q))
    ax.axvline(v, color=INK2, lw=1, ls="--")
    ax.text(v + 5, ax.get_ylim()[1] * 0.9, f"{name}={v:.0f}s",
            color=INK2, fontsize=8)
style(ax, "deliveries", "delivery latency (s)")
save(fig, "m5_latency_hist.png",
     f"30-day latency distribution — {len(all_lats)} deliveries")

lats = all_lats
summary["month"] = {
    "days": len(days),
    "deliveries": int(sum(x["deliveries"] for x in days)),
    "stores": int(sum(x["stores"] for x in days)),
    "dropped": int(sum(x["dropped"] for x in days)),
    "stuck_days": int(sum(1 for x in days if x["stuck"])),
    "leftover_days": int(sum(1 for x in days if x["leftover"] > 0)),
    "p50": round(float(np.percentile(lats, 50)), 0),
    "p95": round(float(np.percentile(lats, 95)), 0),
    "max": round(max(lats), 0),
    "replans_total": int(sum(x["replans"] for x in days)),
    "moves_total": int(sum(x["moves"] for x in days)),
    "rotations_ok": int(sum(1 for x in days if x["rotation_ok"])),
    "evict_mean_s": round(float(np.mean([x["evict_s"] for x in days
                                         if x["evict_s"]])), 0),
    "place_mean_s": round(float(np.mean([x["place_s"] for x in days
                                         if x["place_s"]])), 0),
    "wall_s": round(d["wall_s"], 1),
    "staged_uptime_mean": round(float(np.mean(
        [x["staged_uptime"] for x in days])) * 100, 1),
}

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=1))
