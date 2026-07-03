# Solution design — end-to-end RL for the OOS agent

> The **what** is in [AGENT_BEHAVIOR.md](AGENT_BEHAVIOR.md). This is the **how**: a
> fresh, principled training design built on the clean-slate building blocks. It is
> NOT a port of the prior campaign — that approach is the one the clean slate exists
> to escape (see §8, "What we are deliberately NOT doing").

## 1. Framing — continuing control to minimize cost-to-ideal

The agent runs one shared per-carrier policy (typed-GAT + pointer head) over the
clean-slate `Environment`. The problem is a **continuing control task**: keep the
facility as close as possible to the ideal configuration — *all requests delivered,
all rooms staged, all carriers idle, system maximally retrievable* — under an
adversarial exogenous stream.

We train the underlying *skill* (restore the ideal from any solvable state) on
short episodes that **reset into arbitrary solvable states** (full-state coverage,
§4), because maintenance is just the easy tail of recovery (AGENT_BEHAVIOR §8). A
continuous deployment is then a sequence of these restorations.

## 2. Reward — one cost-to-goal potential, minimal knobs

The reward is the sum of (a) sparse true outcomes and (b) a single
**potential-based shaping** `F = γ·Φ(s′) − Φ(s)` (policy-invariant, telescoping,
un-farmable). We deliberately avoid the prior design's stack of overlapping ladders.

**Potential:** `Φ(s) = −W(s)`, where `W(s)` is an admissible *work-to-ideal*
estimate in carrier-primitive units:

```
W(s) =  Σ_{pending request r}  deliver_steps(r)            # retrieval work
      + Σ_{room m unstaged & not mid-delivery} stage_steps(m)   # staging work
      + λ_tidy · R_excess(s)                               # keep-retrievable (continuous only)
```

- `deliver_steps(r)`: burial-depth + route-hops + handoff + carry + serve — a
  monotone estimate that strictly drops on every productive step (a blocker moved,
  a TAKE, a handoff, a carry roomward, the serve). This is the dense dig breadcrumb.
- `stage_steps(m)`: **room-centric** — 0 if staged, else distance to get an empty
  *docked at that room*. Defined by the room's state, **not** "a carrier holds an
  empty," so a carrier mid-dig holding a blocker-empty never reduces it. This kills
  the prior staging/dig reward conflict *by construction*, with no gating hack.
- **Retrieval-first priority (§7.2)** falls out: while requests are pending, staging
  work for *involved* rooms is excluded (they must un-stage to deliver), and after a
  delivery the room re-stages for free (the served pallet becomes an empty at the
  room). So staging shaping mostly matters post-store and at cold start.
- `R_excess(s)` (continuous mode): the **convex** system-retrieval-cost surplus —
  `Σ_cars convex(dig_cost)` above the floor — so storage placement is shaped toward
  keeping *every* car cheap to dig (AGENT_BEHAVIOR §6.3, the R(s) unification).
  Convex so the worst-buried car dominates (efficiency + solvability protection).

**Sparse true outcomes (the objective; each consumes a task, so un-farmable):**
- `+R_deliver` per requested car actually delivered.
- `+R_clean` one-time on reaching the ideal (episodic anchor).

**Two corrective costs the prior design lacked or got wrong:**
- **Responsiveness** (`−c_resp · max(0, n_unstaged_rooms − n_in_flight_deliveries) · dt`):
  directly encodes §7.2 — un-staging rooms beyond what active deliveries require is
  penalized over time. (The prior design un-staged ~1.6/2 rooms; this is the fix.)
- **No-deadlock** (`−P_dead` when the agent's own action turns a solvable state
  unsolvable, via `_layout_is_solvable`): trains *preserve-solvability* (§10) so we
  do not depend on inference-time escape hacks.

`move_cost` stays a tiny tie-breaker (WAIT-collapse risk; kept ≪ outcomes).

## 3. Observation — add the missing global summary features

The clean-slate obs has `GLOBAL_FEATURE_NAMES = ()`. The constraints live in global
scalars the net currently must reconstruct by pooling (and the value head mean-pools,
averaging away the one deeply-buried SUV). Add explicit globals:
free-slot headroom (total + big-only), max-buried-depth, big-shelf pollution
(non-SUV on SUV shelves), #pending retrieves, #unstaged rooms, fraction staged.
Cheap, and directly serves both the policy's constraint-reasoning and the value
function's worst-case estimate.

## 4. Curriculum & state coverage — decouple training states from the policy

The single most important lever (AGENT_BEHAVIOR §8): the episode-start distribution
must be decoupled from the policy and cover the whole solvable space, forever.

- **Full-state coverage:** every reset draws an arbitrary solvable state
  (`InitialStateSampler` / `shuffle_state` with `require_solvable`) over the full
  hardness space (fullness, big-shelf saturation, disorder, depth, #requests,
  #parked-cars, per-room staged state, direct vs handoff routes).
- **Target-aware reverse curriculum** (new vs prior): a generator that places the
  *specific requested car* at a controlled distance-from-delivered (depth, handoff
  hops, blocker-size mix), and walks the start backward from solved. This converts
  the 20–40-step apex maneuvers from needle-in-haystack discovery into an
  incremental gradient — the thing that makes *pure* RL feasible on the hard tail.
- **Difficulty-frontier weighting:** oversample where success is ~50–80%; prioritize
  resampling of failed/slow states. Advance only on measured greedy mastery.

## 5. Algorithm

Pure PPO (GAE) on the clean-slate net. No imitation (BC/DAgger/solver-oracle all
failed in the prior campaign — covariate shift + multi-agent coordination desync).
Action masking + the policy guards stay. Stability levers to try in order: reward
normalization (`normalize.py`), entropy schedule, KL-regularized fine-tunes to avoid
the catastrophic forgetting the prior small net hit, and **more net capacity** if the
106k plateau (q=2 ~0.96) recurs — pushing the hard tail with *capacity + coverage*,
not harder training on a too-small net.

## 6. Phases

1. **Single retrieve**, full hardness via reverse curriculum → ~100% recovery.
2. **+ store / re-stage / multi-request** (omni clean-rest).
3. **Continuous multi-room stream** — add `R_excess` tidiness + responsiveness +
   latency; SUV admission gate keeps the world always-solvable (§10).
4. **Campus** — train on a topology *distribution* (not single→transfer) for the
   size-invariant coordination the GAT transfer relies on.

## 7. Eval gates — "perfect" is measured, not asserted

A checkpoint passes only if ALL hold (held-out, single-threaded, deterministic):
1. **Recovery** — ~100% success on a held-out hard battery across the full spectrum
   (incl. apex SUV), + steps-to-restore vs optimal.
2. **Maintenance** — continuous stream: rooms-staged fraction, retrieval latency,
   throughput, travel-per-task.
3. **Responsiveness** — measured `n_unstaged_rooms` while a retrieve is pending
   tracks in-flight deliveries (≤ 1 per active delivery); the prior ~1.6/2 is a fail.
4. **Solvability-preservation** — `_layout_is_solvable` every step; any
   solvable→unsolvable transition is a hard fail.
5. **Never-stuck** — loop/cycle detection + timeout; zero tolerance — **and met by
   the policy alone, with no inference-time escape hatch.**
6. **Generalization** — campus zero-shot + few-shot drop.

## 8. What we are deliberately NOT doing (lessons as guardrails)

The prior campaign reached battery 1.0 but was flawed; we keep its lessons, not its
code:
- **No inference-time crutches** (cycle-escape, MCTS-on-cycle). Robustness must be in
  the policy, driven by coverage + the no-deadlock / responsiveness signals. If the
  policy loops, that is a training failure to fix, not a runtime patch.
- **No post-hoc action-mask guards** to paper over learned-policy gaps (the prior
  needed 4). A guard that encodes a true physical rule (e.g. never GOTO a shelf you
  can't give to) is fine; guards that substitute hand-logic for missing skill are not.
- **No 30-knob reward** with overlapping mechanisms and a gated staging/dig conflict.
  One potential, a few clear terms.
- **No BC / DAgger / solver-oracle** — proven dead ends here.
- **Do not push the hard tail with harder training on a too-small net** (caused
  catastrophic forgetting). Use coverage + capacity + reverse curriculum.

## 9. Status

Foundation validated on the clean slate: the PPO pipeline composes
(env→net→rollout→update→checkpoint→load) and the `InitialStateSampler` produces
controllable, 100%-solvable states across the hardness sweep. Building §2–§4 next.
