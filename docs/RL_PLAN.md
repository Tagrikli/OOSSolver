# OOSKiller RL Campaign Plan: Continuous ASRS Control of tiny_medipol

Reference design-doc for the campaign to train a VERIFIED RL agent that runs the
tiny_medipol facility, minimizing expected per-task WAIT time, and — the central
requirement — handles structured HARD configurations (deep target, tight
eviction-slack, buffer-on-target), not just the easy random middle.

---

## 1. The objective and the target behavior (derived, not shaped)

Minimizing expected per-task WAIT = the integral of pending-task age over time.
Its unique optimum at tiny_medipol is the **clean-resting state**: every lift
(room) carrier parked at its room holding an EMPTY pallet (staged), every shuttle
EMPTY, every carrier WAITing. On a store: store + re-stage. On a retrieve: dig +
deliver, then settle. This is exactly `RetrieveEnv`'s omni success predicate
(`_park_all_staged` ∧ `_all_targets_delivered` ∧ `_noroom_carriers_empty` ∧
`_all_carriers_waiting`, retrieve_env.py:940-945). We do not hand-shape this
behavior; we anchor on it with a one-time terminal reward and let the optimum
emerge.

## 2. Difficulty characterization — axes, construction, measurement

**Axes** (the hardness coordinates the uniform `shuffle_state` sampler
under-covers, causing the 98%-random / 0%-hard split):

| axis | symbol | values | knob |
|---|---|---|---|
| burial depth | d | 0,1,2 (cap-3 shelves) | constructive / `depths` |
| blocker bigness | B = bigs_in_front(target) | 0,1,2 | constructive |
| **eviction-slack** | **K = free_other_bigs − bigs_in_front** (on the TARGET) | >=2, 1, 0, **−1** | constructive |
| routing | direct vs handoff | {False,True} | `target_any_shelf` |
| #simultaneous tasks | k room cars, q requests | (0,1,2),(1) | `room_car_amounts`,`request_car_amounts` |

**Critical correction (from review).** `K` must be computed on the **actual
requested target**, via `free_other_bigs(shelf)` and `bigs_in_front(shelf,
target_idx)` (refactor these out of `shuffle._layout_is_solvable`'s inner
closures, shuffle.py:241-256, into shared module-level helpers). The sampler's
`_pick_representative` (shuffle.py:296) computes `K` on a per-shelf worst-case
pallet, NOT the dug one — gating on it trains the easy middle relabelled.

**The benchmark region is K <= 0.** `_layout_is_solvable` accepts a layout only
when `bigs_in_front(target) <= free_other_bigs` (shuffle.py:287) — the *negation*
of buffer-on-target. So `require_solvable=True` (and `InitialStateSampler`, which
repairs bigs→empties shallowest-first when it can't meet the bound,
state_sampler.py:208) **structurally cannot emit the benchmark**. The K<=0 region
must be built CONSTRUCTIVELY.

**Construction.** Use the EXISTING `RetrieveEnv.set_forced_layout(builder)` hook
(retrieve_env.py:431 — "Used by the hard-case battery and the curriculum to
inject exact, difficulty-controlled configurations"). `make_buffer_on_target_layout`
places the requested big at depth `d` with exactly `B` bigs above it, fills the
OTHER big shelves so `free_other_bigs = B + K`, leaves >=1 free slot on the target
shelf (so the put-back is physically possible), and seeds the Retrieve via
`env._seed_retrieve`. Assert realised `target_slack == K` on the built layout.

**Measurement.** A HELD-OUT battery: per-tier fixed-seed forced layouts on a seed
range disjoint from training (>=32/tier). Evaluate **greedy (argmax) only**
(greedy_collapse_sampling_noise: stochastic rate is entropy noise). Assert each
eval layout's realised (d,B,K) before counting it. Report per-tier and a 2D
heatmap over (d × signed-K) — the failure mode is a cold high-d/negative-K
corner.

## 3. The real blocker: hard EXPLORATION, not coverage or density

The benchmark solve REQUIRES temporarily GIVE-ing a blocker onto the **target's
own shelf** to free another big shelf as buffer
(buffer_on_target_scenario.md). The action model permits this — `GIVE` is legal
on any docked shelf passing size/capacity (action.py:166), with no mask against
the target shelf. The blocker is that this move's probability "stays vanishingly
small — exploration can't find the success once, so credit assignment never
reinforces it."

This has a decisive consequence the reviewers confirmed: **any potential monotone
in burial depth fights the solution.** The put-back move raises
`bigs_in_front(target)`, so Φ = −steps_to_rest (pbrs design) or the base
`_retrieve_remaining` depth+3 ladder (env.py:650) both pay `F < 0` on the one
required action. Curriculum (sample-efficiency), PBRS (credit-density), and PLR
(level-distribution) are all orthogonal to surfacing a near-zero-probability
action. The ONE family of mechanisms that helps: **an exact solver that can
demonstrate the move (imitation), or an intrinsic novelty bonus that raises its
sampling probability** (RND / count-based), as flagged in both memories.

## 4. The six candidate designs (with verdicts)

### A. Resting-Potential PBRS (`reward_pbrs`) — verdict: hygiene, wrong target
One-time terminal +20 at clean-rest, shaped by a single non-farmable Φ =
−steps_to_rest. Cleanest reward, provably unfarmable, kills WAIT-collapse if
ProgressTerm is used. **Fails the benchmark**: Φ monotone in depth → `F<0` on the
put-back. Its curriculum assumes a continuous env that is deleted
(`set_auto_arrivals(False)`, retrieve_env.py:441). Salvage: the reward discipline.

### B. SLACK-Ladder Curriculum (`curriculum`) — verdict: BEST chassis
Swap `shuffle_state` for tier-indexed construction; escalate on greedy mastery of
a held-out frontier; rehearse cleared tiers. Minimal reward (terminal + one PBRS).
Two fixable flaws: (1) gates through `require_solvable` (excludes K<=0); (2)
monotone Φ fights the put-back. Fix (1) with constructive K<=0 layouts via the
existing `set_forced_layout`; fix (2) by keeping Φ depth-agnostic and adding an
exploration bonus at hard tiers. **Only L-effort design whose core mechanism is
already buildable on the live env.**

### C. ORACLE-DAgger (`imitation_search`) — verdict: strongest ceiling, XL, wrong-first
Exact branch-and-bound solver over sim primitives → dense per-state labels →
DAgger → optional PPO polish. The ONE design that can DEMONSTRATE the put-back,
dissolving the exploration wall. But: cost must be the concurrent **sim clock**
(`state.time`), not a sum of command durations (4 carriers run in parallel); the
heuristic must be **max-over-carriers (critical-path)**, not a serial sum, to stay
admissible; the greedy fallback must be dropped from training labels (it is the
known-failing heuristic on the benchmark cell). The whole bet reduces to: can the
solver crack the single buffer-on-target config? Prove that in isolation FIRST.

### D. PLR-Regret (`plr_regret`) — verdict: accelerates easy→medium, misses the wall
PVL regret prioritizes high-value-loss levels. But PVL ≈ 0 on a never-almost-
solved level (reward≈0 → GAE≈0 → de-prioritized), so it never surfaces the
benchmark; the slack grid {0,1,2,3} is on the wrong side of zero; rollout
fragmentation (256 transitions/env vs 1024-step truncation) breaks per-episode
scoring. A distribution method for an exploration problem.

### E. Continuous wait-cost (`continuous_avgreward`) — verdict: right objective, wrong first move
A never-resetting StreamEnv optimizing the per-task-WAIT integral IS the true
deployment objective. But it needs infra that is deleted (StreamEnv, Poisson
wiring), missing observations (`GLOBAL_FEATURE_NAMES=()`, no task-age → reward is
non-Markovian), an average-reward GAE path, and it fights the engine's dt
fast-forward of idle instants. Defer to the deployment phase.

### F. HOOM (`hierarchical`) — verdict: right idea in unbuildable scaffold
The target-aware curriculum inside it is correct; the hierarchy is not. The
learned task→carrier router is not an env decision point (the env queries one
carrier for a primitive); the continuous router phase needs deleted infra;
per-token Φ swaps break PBRS telescoping at option seams. Cut the hierarchy, keep
the curriculum.

## 5. Recommended plan — Curriculum chassis + exploration injection

**Build B (curriculum), fixed and hardened, FIRST.** It is the only L-effort
design whose coverage attack is cheap and bug-resistant on the live episodic env,
and it is a strict subset of the work every other design needs (they all need the
forced-layout hard-case construction). It de-risks the DAgger decision: if the
novelty-bonus PPO cracks the held-out K<=0 battery, DAgger is unnecessary; if it
stalls, the hard-case solver harness built for the battery is exactly the
bootstrap DAgger needs.

### Reward (anchor + one shaper + rescue + hard-tier exploration)
- **REWARD_SUCCESS = +15.0**, once, on clean-rest. Terminates → unfarmable; GAE
  zeroes V(s') (rollout.py:337) → lands undiluted. Reused from train_omni2.
- **Shaping = ProgressTerm** (reward_system.py:373), `F = Φ(s')−Φ(s)`,
  un-discounted → a no-op pays exactly 0 → WAIT-collapse impossible. NOT
  PotentialTerm (γ<1, Φ<=0 → positive idle-drip over MAX_EPISODE_STEPS=1024).
- **Φ = binary-holds ladders only** (retrieve_env.py:1069,1077): retrieve rung
  shuttle-holds 1.0 < room-holds 2.0; staging rung empty-handed 1.0 / empty 2.0 /
  staged 3.0. Total height ~5 < SUCCESS/2. **No depth term, no slack term** — Φ is
  deliberately agnostic to HOW the dig is achieved, so it does not fight the
  put-back.
- **AllWaitWhileTaskTerm penalty = 5.0** as the stall-rescue ONLY (wake +
  re-query). Guard: check success BEFORE the rescue wakes carriers, so it does not
  race the `require_all_waiting` latch.
- **NoveltyBonusTerm (T>=8 only)**: `+beta/sqrt(1+N(key))`, key = (target-shelf-id,
  GIVE-onto-target-shelf), beta=1.0 (< SUCCESS/10), N monotone per-run. Decays →
  non-farmable; the env already masks immediate-inverse and reverse-GOTO bounces
  (action.py:97-103,144-150). This is the mechanism that raises the put-back's
  sampling probability enough to be found once.
- MOVE_COST = 0; all event-reward and r_car2* weights = 0 (anti-farm: the +46/0%
  pathology is the event stack).

### Curriculum (13 tiers, greedy-mastery-gated, rehearsal-mixed)
Tiers T0..T12 ordered cheap-axes-first, lethal negative-K last (full table in the
campaign's curriculum spec). T0 = target on top (the +15 found in 1-2 argmax
actions, bootstraps everything). T8 = first K=0 tier (NoveltyBonus on). T10-T11 =
K=−1 buffer-on-target. T12 = the pinned medipol fullness=1.0 config. Easy tiers
use `shuffle_state`; hard tiers (T>=8) use constructive forced layouts.

Promotion: greedy(T_cur) >= 0.90 on the held-out battery for 2 consecutive evals
(debounce). Never auto-demote; pause + bump rehearsal weight if a cleared tier
< 0.60. Rehearsal: each reset draws T_cur with prob 0.5, else uniform over cleared
tiers; >=0.15 floor on every opened tier. MAX_EPISODE_STEPS raised to 96-128 for
hard tiers (the put-back is longer than the naive dig — a 64-cap forbids it).

## 6. Closed-loop methodology (how we know it works)

1. **Anchor sanity**: a scripted solve of a constructed K=−1 config reaches the
   clean-rest predicate (success is *reachable* before we ask PPO to find it).
2. **Anti-farm**: 200 forced-WAIT + 200 random-loop episodes all return < 15.
3. **No-op = 0**: ProgressTerm pays exactly 0 on a no-op step.
4. **Φ does not fight the maneuver**: assert the put-back GIVE does NOT lower the
   *binary-holds* Φ (it doesn't — Φ ignores depth).
5. **The metric**: held-out greedy clean-rest success PER TIER, especially T10-T12
   (K<=−1) — target >=0.80 argmax. The (d × signed-K) heatmap cold corner warms.
6. **Forgetting guard**: T0/T1 greedy never drops > 0.10 after promotion.
7. **Promotion health**: T_cur is a monotone staircase vs iters; a long plateau at
   T8-T12 with the heatmap corner still cold ⇒ the novelty bonus is too weak OR
   the maneuver needs imitation — escalate to ORACLE-DAgger (design C), reusing the
   battery's hard-case construction as the solver's input set.
8. **Deployment (later phase)**: once the puzzle is solved episodically, move to
   the continuous wait-cost objective (design E) with task-age observations added
   first — the curriculum-cooked brain warm-starts it.

## 7. Failure-mode ledger (each closed by construction)

- **Coverage (#1, central)**: constructive K<=0 layouts measured on the real
  target + greedy-gated held-out battery → the benchmark region is trained AND
  measured, never averaged away.
- **Farming (#2)**: only non-telescoping term is the one-time terminal that ends
  the episode; all shaping is ProgressTerm (no-op=0, telescopes); novelty decays.
  Max non-success return < 15.
- **WAIT-collapse (#3)**: no flat per-step cost anywhere; ProgressTerm no-op=0;
  the all-wait penalty is a rescue trigger, not the gradient.
- **Sparse-terminal (#4)**: T0 makes the +15 a 1-2-action discovery; each
  promotion adds one increment over a mastered policy; ProgressTerm densifies
  within-episode; the novelty bonus surfaces the one needle action.
- **Forgetting (#5)**: rehearsal mix over all cleared tiers with a >=0.15 floor +
  demotion guard.

## 8. Biggest residual risks

- **The put-back stays unsampled even with the novelty bonus.** If beta high
  enough to surface it also destabilizes easy tiers, the bonus is too blunt — then
  the maneuver needs *demonstration*: escalate to ORACLE-DAgger (C), which is why
  the hard-case construction harness is built first (it is the solver's input).
- **Rescue/latch race**: the all-wait wake can knock the facility out of the exact
  resting state the +15 demands — guarded by checking success before firing the
  rescue; covered by test (c).
- **Constructive builder correctness**: assert realised (d,B,K) on every built
  layout, training AND eval, so a silent off-by-one never eases the benchmark.
