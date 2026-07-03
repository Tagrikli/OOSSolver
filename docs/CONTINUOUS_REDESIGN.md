# Continuous-operation redesign (deferred — try later)

> Status: **proposed, not yet implemented.** Captures a diagnosis of why the current
> recovery checkpoint (`runs/best/oos_agent.pt`) isn't *fluent* in continuous
> operation, and a redesign to fix it. Pick this up when ready to spend compute.
> Companion docs: [AGENT_BEHAVIOR.md](AGENT_BEHAVIOR.md) (the target behavior),
> [SOLUTION.md](SOLUTION.md) (what was actually built).

## 0. Where we are

The shipped model (`runs/best/oos_agent.pt`, the v6 recovery checkpoint) is strong on
the **hard skills**: 0.99 recovery success from arbitrary solvable states, **0
deadlocks**, ~0.97 throughput in continuous deployment. But watching it run live, it
is **not fluent**. Four observed symptoms:

1. **Irrelevant carriers un-stage rooms** they have no role in the current task.
2. **Multiple simultaneous requests** confuse it — it can't reliably solve them.
3. **High-SUV park** can't recover even when SUV slots and a solution exist.
4. **Non-serving carriers never stop** — with no task they wander, go-and-come-back
   with no goal.

## 1. Diagnosis (root causes, ranked)

The unifying cause: the current reward optimizes the **outcome** ("reach clean-rest,
then the episode *ends*") but barely shapes the **process** (every carrier doing
something useful or resting). Two structural choices make this concrete: episodes
terminate the instant clean-rest is reached (so the policy never trains the
*continuing*/at-rest regime), and the potential Φ is **silent on task-irrelevant
carriers** (no gradient telling them to stay still).

| Symptom | Primary cause | Notes |
|---|---|---|
| 1 (un-stage), 4 (wander) | **Reward shaping + training regime** | Outcome reward only says "reach clean-rest" — a *sloppy* recovery scores the same as a clean one. No idle-discipline signal; rest behavior never trained (episodic termination). NOT capacity, NOT curriculum. |
| 3 (SUV park) | **A missing reward term** | The "keep-retrievable" potential `R_excess(s)` designed in SOLUTION.md §2 was never implemented, so storage *quality* (route SUVs to SUV shelves, keep them clean, handoff-store) is unshaped. Plus SUV-pressure is rare in the curriculum. |
| 2 (multi-request) | **Curriculum under-coverage + symmetric reward** | Φ = −(total work) is symmetric over requests, so a greedy per-carrier policy piles both carriers on one task or contends at handoffs (→ loops). Multi is the rarest/last-trained case. Capacity is a *secondary* contributor here only. |

Key insight: a proper PBRS potential is **policy-invariant** — the shaping is NOT what
makes it un-fluent. The un-fluency lives in the *outcome* reward (no fluency
incentive) and the *regime* (episodic). So the fix is not "remove shaping"; it's
"make the true objective the reward, and train the continuing regime."

## 2. The redesign — minimize a running cost, forever

Reframe from **episodic "reach the goal then stop"** to **continuing "keep the system
at the ideal, forever"**, where the agent minimizes a **per-second cost rate**. This
is literally the deployment task, so the wanted behaviors fall out of the objective
instead of being bolted on.

### Reward = −cost, charged each decision step (elapsed `dt`):

```
cost =  w_wait    · Σ_pending (per-task wait)        · dt   # unserved customers cost time
      + w_unstage · (rooms un-staged beyond in-flight need) · dt   # responsiveness (§7.2)
      + w_idle    · (movement by carriers with NO role in any active task)   # idle discipline
      + w_tidy    · R_excess(s)                       · dt   # keep-retrievable / storage quality
      + P_dead    · [a move made the state unsolvable]       # never deadlock (one-off)
   −  B_deliver   · (cars delivered this step)               # outcome bonus
   −  B_store     · (cars stored+room re-staged this step)   # outcome bonus
```

Maximizing reward = keeping the system served, responsive, tidy, fluent, and solvable.

### Why each symptom is fixed

- **1 & 4 (un-stage / wander):** `w_unstage` and `w_idle` directly fine needless
  un-staging and aimless motion → idle carriers learn to sit still. **Critical:**
  `w_idle` is **conditional** — only carriers with *no role* in any active task pay
  it (a carrier is "active" if it holds/digs a requested pallet, its room is a chosen
  delivery target, or it's the designated empty-fetcher). Conditional is what avoids
  the WAIT-collapse my earlier *blanket* `move_cost` caused.
- **2 (multi-request):** each pending task bleeds `w_wait` **independently**, so
  ignoring the second task is strictly costly → forces the agent to split carriers
  across tasks. (This is the big advantage over the old symmetric Φ.) Pair with heavy
  multi-request exposure in training.
- **3 (SUV park):** `w_tidy · R_excess` is the keep-retrievable signal — pushes SUVs
  onto SUV shelves and keeps them clean. Also shape the **handoff-store** maneuver
  (storing a big car that needs a handoff to reach a free big shelf), which is
  currently unshaped like the cold-handoff-staging loops.
- **rest at idle:** when nothing is wrong every fine is 0 and every move costs, so
  **WAIT is strictly optimal** — resting is not a separate trained mode, it's just
  "don't disturb a zero-cost state."

`R_excess(s)` = convex system-retrieval-cost surplus: `Σ_cars convex(dig_cost)` above
the floor (depth + handoff distance + SUV-shelf-pollution), so the worst-buried car
dominates (efficiency + solvability protection). See SOLUTION.md §2.

## 3. Training recipe — add fluency WITHOUT degrading the dig skill

The earlier continuous fine-tunes **degraded** recovery because they trained on the
new objective *only*, with a reward that fought itself (staging-always vs serving; an
anti-cycle bug that penalized standing still) and started from random states (so they
practiced recovery, not maintenance). Avoid all of that:

1. **Warm-start from `runs/best/oos_agent.pt`** — the 0.99 dig/handoff/no-deadlock
   skill is good; keep it.
2. **Mix the training distribution** in every batch:
   - ~50% **recovery rehearsal** (the existing reverse curriculum + coverage) so the
     dig skill keeps getting gradient and can't drift — this is the anti-degradation
     anchor.
   - ~50% **continuous-stream episodes that DO NOT terminate at clean-rest**, started
     from a mix of clean (practice maintenance) and messy (practice recovery) states.
3. **Low LR** (e.g. 1–1.5e-4) and optionally a **KL leash** toward the warm-start, so
   it's a nudge, not an overwrite.
4. **Reverse curriculum stays as the learnability scaffold** (a deep dig only pays off
   on completion, so it's still sparse-ish). Optionally **anneal the potential shaping
   to zero** as the curriculum opens, so the final policy is judged purely on the
   true cost-rate objective (no shaping artifacts).
5. **Heavy multi-request emphasis** in both the curriculum and the stream.

## 4. Caveats / what to tune

- **Weight tuning is the main risk.** Too-heavy `w_idle` reintroduces WAIT-collapse
  (agent freezes even when it should move). Start the fines small, watch the viz,
  raise gradually. The conditional gating on `w_idle` is what makes it safe.
- **"Active carrier" predicate** (who pays `w_idle`) must be computed carefully — get
  it wrong and you either tax productive carriers or let wanderers off free.
- **Capacity** is still the residual ceiling for the apex 2-deep-multi case; only bump
  net size (hidden 128 / gat 3, the `recovery_big` config) AFTER the reward/curriculum
  are right — capacity is the *last* lever, not the first.
- **Watch it, don't just trust aggregate metrics.** The shipped model's continuous
  aggregates (0 deadlocks, ~97% throughput, ~80% idle-staging) looked fine yet masked
  the wandering/un-staging — fluency needs eyeballing + a redundant-motion metric.

## 5. Implementation pointers (where things live)

- Reward + env: extend `oos/env/recovery_env.py` (it already has `continuous`,
  `c_resp`, `p_deadlock`, conditional pieces partly wired; `_state_hash` for cycle
  logic; the anti-cycle is already gated to only fire when work remains). Add
  `R_excess`, the conditional `w_idle`, the per-task wait cost, and the
  "active-carrier" predicate.
- Trainer: `oos/learn/train_continuous.py` exists (clean-start + idle-only staging +
  continuous eval already in place); needs the recovery-rehearsal mix added (borrow
  the env-population logic from `oos/learn/train_recovery.py`).
- Eval: `oos/learn/continuous_eval.py` (throughput/latency/staging/deadlocks) +
  `oos/learn/recovery_eval.py` (recovery gate) — run BOTH so we confirm fluency
  improved without recovery degrading.
</content>
