# Solution v2 — the pallet-move semi-MDP

> Status: **proposed design, synthesized 2026-07-02**. Successor to
> [SOLUTION.md](SOLUTION.md) (episodic recovery + PBRS, trained; strong dig skill,
> not fluent) and [CONTINUOUS_REDESIGN.md](CONTINUOUS_REDESIGN.md) (cost-rate
> reframe, tried in 9 fine-tunes + 1 from-scratch run; plateaued). The system
> ([PROBLEM.md](PROBLEM.md)) and the target behavior
> ([AGENT_BEHAVIOR.md](AGENT_BEHAVIOR.md)) are unchanged and immutable.
>
> Provenance: produced from a multi-agent design review — 4 codebase readers,
> 5 independent design proposals (action abstraction / search & model /
> objective & regime / architecture & credit / contrarian-minimal), each
> adversarially critiqued from two lenses. This document keeps what survived.

## 0. TL;DR

Stop training the policy to choose **carrier primitives** (GOTO/TAKE/GIVE/WAIT
every ~3 s). Train it to choose **pallet moves** — `MOVE(source pallet →
destination)` — and compile each move to primitives with a small deterministic
executor. Everything the evidence says is broken lives in the primitive
granularity, and everything the evidence says already works (the dig skill,
the sim, the typed-GAT, the curriculum machinery) carries over one level up:

- **Wandering and needless un-staging become inexpressible** — there is no
  move-level action that moves a carrier without relocating a pallet. The two
  fluency symptoms that nine fine-tunes of sub-noise fines (`w_idle=1e-6/mm`,
  the `w_idle_fixed=0.1` "survives reward-normalization noise" hack) could not
  buy are deleted from the action space instead of priced.
- **Rendezvous desync becomes impossible** — one move authors every involved
  carrier's script; coordination is compiled, not learned.
- **The horizon collapses ~5×** — a depth-0 retrieval is 2–3 moves, the §9 apex
  SUV maneuver ~6–8 moves instead of 25–40 primitives. From-scratch PPO becomes
  feasible again (the tiered run died at tier 0 because a depth-1 dig is a
  ~10-primitive needle), so the whole warm-start/KL-leash/forgetting apparatus
  is retired.
- **Never-deadlock becomes an invariant, not a trained property** — every move
  is filtered through a solvability-preserving mask backed by an *exact*
  move-simulating check (not the current big-shelf heuristic), and admission
  control is on, environment-side, always.
- One **seconds-denominated continuing objective** (pending-task bleed +
  responsiveness + tidiness potential + move time), identical from the first
  episodic stage through continuous deployment — no objective switch, no
  reward normalizer, SMDP-correct discounting. Fines are O(1) per decision,
  not O(dt·1e-6) against normalizer noise.

Deployment artifact: the trained policy, greedy, policy-only. Search appears
only offline (curriculum certification, eval auditing, optional
expert-iteration teacher if the apex tail stalls).

## 1. Diagnosis — what the evidence actually says

The empirical record (runs/, docs/plots/, all verified in code):

| Fact | Evidence |
|---|---|
| The dig/execution skill is **solved** | latency boxplot: 100 % greedy success at depths 0/1/2 (n=600), medians 45/70/97 s, handoff routes no slower than direct, 0 deadlocks anywhere |
| Maintenance fluency is **stuck** | 9 warm-start+KL fine-tunes: staging uptime plateaued 0.43–0.57 (gate 0.90), redundant motion 0.10–0.15 (gate 0.02), entropy drifting *up*; recovery oscillating 0.94–0.993 (one run degraded 0.993→0.940) |
| From-scratch primitive RL **can't discover** the task | tiered run: tier 0, success 0.000 after 22 iters; dig never found |
| The un-staged time is **in-flight work**, not idling | resting-rate already 0.014–0.042 across all failed runs — carriers are *busy doing inefficient things*, so more idle-fines were never the answer |

Root causes, each traceable to a mechanism:

1. **Granularity mismatch (the big one).** At ~3 s primitive ticks, every
   fluency objective (rest, §7.2 responsiveness, §7.3 tidiness) must be
   expressed as per-step fines of 1e-6…0.1 that then pass through per-minibatch
   advantage normalization and a running-return `RewardNormalizer`. The code
   itself confesses it: `w_idle_fixed` exists "so WAIT is the robust argmax"
   against "reward-normalization noise" (recovery_env.py). The reward channel
   structurally loses that SNR fight; nine sweeps confirmed it.
2. **Exploration cliff.** A depth-1 dig is ~10 coordinated primitives across
   2 carriers before any outcome signal. Reverse curriculum + warm start
   papered over it; from scratch it is fatal.
3. **Learned coordination.** Two-carrier rendezvous timing had to be *learned*
   (and killed BC/DAgger via desync). A symmetric potential piled carriers onto
   one task.
4. **Observability hole.** A moving carrier's destination is invisible
   (`docked_none` in transit; no committed-target feature) — multi-request
   coordination was partly *unlearnable* regardless of reward.
5. **Accounting bugs.** Fixed γ=0.99 per *variable-dt* decision ⇒ ~300 s
   effective horizon (burial costs paid at request time, minutes later, are
   discounted to invisibility). `RewardNormalizer(n_envs=1)` shared across 14
   vec envs interleaves one return accumulator. Mean-pooled value head averages
   away the one ignored request / one buried SUV.
6. **Spec drift.** `stage_steps` is carrier-based, not room-centric (the
   staging/dig conflict SOLUTION.md §2 claimed to kill by construction is
   alive); `_r_excess` is linear, not convex; `c_resp` counts pending retrieves,
   not in-flight deliveries.
7. **Hygiene violations.** `shuffle_state(require_solvable=True)` **silently
   accepts an unsolvable layout after 200 retries** (§10 poison);
   `InitialStateSampler`'s big_ratio/disorder axes (the §8 coverage lever for
   the SUV symptom) are wired to no trainer; `_layout_is_solvable` is a
   big-shelf-only heuristic that is *neither sound nor complete* as a
   solvability oracle (ignores carrier-held pallets, reachability, small-shelf
   capacity); `_big_admission_ok` has a concurrency race (an admitted big still
   riding a carrier is invisible to the next admission check).

Conclusion: the two prior designs kept the primitive action space and tried to
fix behavior with reward engineering. The behaviors that failed are exactly the
ones that space makes either sub-noise (fluency), needle-in-haystack
(discovery), or unlearnable (coordination, without intent observability). The
fix is to change the *decision space*, not the knobs.

## 2. The action space: `MOVE(src, dst)` + `HOLD`

The facility's true decision problem is pallet-flow combinatorics — *which
pallet goes where, in what order*. Routing, rendezvous, and timing are
deterministic mechanics the sim already computes in closed form (trapezoidal
`travel_time`, auto-handoff on rendezvous, BFS handoff routes). So:

**Action** = `MOVE(source, destination)` or `HOLD`, chosen by one centralized
policy at **move-level decision epochs**.

- **Sources**: top pallet of each shelf; pallet held by each unclaimed carrier.
  (tiny_medipol ≤ 20, campus ≤ 150.)
- **Destinations**: any shelf with (reservation-adjusted) free capacity; any
  room. (tiny_medipol ≤ 18, campus ≤ 145.)
- **Room semantics fall out uniformly**: dst=room with an empty pallet = *stage*;
  dst=room with the requested car = *deliver*; src=the staged empty on a
  room-docked carrier = the *explicit un-stage decision* (one attributable
  action, not a diffuse primitive habit). Store = src=parked-car pallet on the
  serving carrier, dst=some shelf. Tidying = shelf→shelf. Everything §6 of
  AGENT_BEHAVIOR requires is a composition of these.
- **Decision epochs** (semi-MDP): whenever an exogenous event fires or a
  carrier is freed, while ≥1 startable move exists; multiple emissions at the
  same instant proceed sequentially at dt=0 until HOLD or no free carrier
  (mirrors today's `pending_idle` pattern one level up).
- **HOLD** is masked when work is pending and *nothing is in flight* — with the
  world solvable (§3) a productive move always exists then, and this closes the
  livelock hole where a HOLD with an empty scheduler would hang the clock.

**The executor** (`MoveExecutor`) compiles a chosen move into per-carrier
primitive scripts (GOTO/TAKE/GIVE, handoff-pose rendezvous via the sim's
existing auto-handoff), picking the route/relay by min analytic ETA:

- **Atomic claims, free carriers only (v1).** A move is startable iff *all*
  carriers on its route are currently unclaimed and idle. No queuing claims on
  busy carriers — this trivially excludes circular-wait deadlock (the FIFO
  claim-queue variant admits a textbook cycle on campus; pipelining/eager
  second-leg dispatch is a later optimization that must ship with a proven
  global claim order).
- **Closed-loop execution.** Every primitive re-checks engine preconditions
  before submission; a full-duration **reservation ledger** (per-shelf pending
  give/take for the whole move, not just while docked) guarantees GIVE/TAKE
  legality at execution time and makes concurrent moves collision-free.
- **Non-preemptible in v1** (one pallet relocation, ≲30 s worst case). Measured
  against the 45 s depth-0 latency floor; if the reactivity tax shows up in
  evals, the prepared extension is ABORT-before-TAKE at leg boundaries.
- The env layer is a **shim** (`MacroEnv`): when the base `Environment` queries
  a carrier for a primitive, the shim answers from that carrier's active script
  (or WAIT). `oos/sim` is untouched; the primitive path stays for viz/planner.

**Expressiveness audit** (what is lost vs primitives, and why that's
acceptable): every pallet-configuration change factors through pop→carry→push,
so any rearrangement plan is a composition of moves. Genuinely lost: (a)
anticipatory empty-carrier pre-positioning against *future* arrivals — which
AGENT_BEHAVIOR §2 explicitly declares worthless ("the agent earns nothing by
anticipating either distribution"); (b) carrier-as-buffer plays (a partner
holding a blocker in the air while the digger works). (b) is real but narrow —
it only matters in extreme stranded-air states; if the eval battery ever shows
it, the extension is a `PARK(src → handoff pose, hold)` move variant. Neither
loss threatens §9's apex cases, which are pure move sequences.

## 3. Never-stuck, layer by layer (and what each layer really guarantees)

**Layer 1 — the world is always solvable (invariant, provable).**

1. *Solvable initialization*: `shuffle_state`'s silent unsolvable-accept after
   200 retries is replaced by repair-or-reroll (the `InitialStateSampler`
   already has the deterministic repair). Never emit an unsolvable start. (§10)
2. *Admission control on, environment-side, always* (AGENT_BEHAVIOR Q3's own
   recommendation): `gate_big_retrievability=True` in training and deployment —
   **fixed for concurrency**: the admission check must count bigs that are
   in-flight (admitted but still riding a carrier / reserved to a slot), or two
   near-simultaneous big stores can jointly overrun the last big slot.
3. *Solvability-preserving move mask*: a move is legal only if the post-move
   state is solvable. Three hard-won specification points (each was a fatal
   flaw in a draft of this design):
   - The check must **simulate the full move** — pop the source *and* push the
     destination. A memoized placement-only check is unsound: there are §9 apex
     states whose only solution's first move passes the true check but fails
     the no-pop version (the pop is what frees the closure). Monotone
     memoization is kept **as a fast path only**: memo-pass ⇒ sound-pass, but
     memo-fail ⇒ run the exact check before masking.
   - The oracle must **account for carrier-held pallets and in-flight move
     destinations**, not just shelf stacks. The current `_layout_is_solvable`
     sees neither and is simultaneously too optimistic (assumes smalls/empties
     always have somewhere to go; ignores reachability) and too pessimistic
     (statically rejects legitimate apex transients that spend the last big
     slot while an in-flight delivery is about to free capacity).
   - Therefore the mask is backed by a real **move-level solvability oracle**
     (`oos/plan/oracle.py`): "every stored car has a feasible dig plan in the
     move graph, given current stacks + carried pallets + reservations." At
     tiny_medipol scale this is an exact bounded search (milliseconds; moves,
     not primitives). For campus it runs incrementally/regionally with a
     *proven-conservative* fast path and exact recheck on fast-path failure —
     so the mask never wrongly forbids the apex maneuver (a conservative-only
     mask silently reintroduces a difficulty ceiling, violating §8).

With 1–3, every issued Retrieve has a solution at every instant, forever. The
prior design *trained* this property (`p_deadlock`, two solvability checks per
step, half the throughput); here it cannot be violated, and `p_deadlock` is
deleted.

**Layer 2 — executor liveness (provable).** Atomic all-free claims exclude
claim cycles; scripts are finite with closed-form durations; rendezvous cannot
desync because one move authors both partners' scripts and the sim's
auto-handoff fires on co-presence; the reservation ledger guarantees no script
stalls on a rejected primitive. Every emitted move completes in bounded time.

**Layer 3 — policy liveness (trained, gated, honestly empirical).** Whenever
work is pending, solvability guarantees the mask offers a productive move; the
HOLD mask closes the idle-livelock; every pending task bleeds cost so loops are
value-dominated; a small anti-cycle term fires on *physical-state-hash* revisit
while work is pending (hash excludes commitments, so commitment-churn cannot
whitewash a physical loop). We do **not** claim a theorem here: the policy
could still ping-pong. It is gated to zero over the eval batteries (§7), and —
the key hygiene upgrade — the **oracle audits every failure**: because it
searches the exactly-masked move space *and* the unmasked space, it can
distinguish "policy gap" from "mask incompleteness" from "genuinely unsolvable
task", so a training signal is never poisoned and a mask bug is indicted
rather than mislabeled as an impossible request.

**Layer 4 — optional deployment watchdog (disclosed, default off).** A
W-progress monitor with a solver fallback would make never-stuck a theorem at
deployment, but contradicts the standing no-inference-crutch guardrail. Ship
decision is deferred; either way every gate requires the watchdog (or its
monitor-only variant) to fire exactly zero times.

## 4. Objective — one continuing cost rate, in seconds, for every stage

Per move-level transition with elapsed sim-time τ:

```
r = − [ 1.0·n_pending_retrieves + w_store·n_pending_stores ] · τ     # latency bleed (§7.1); per task, independent
    − c_resp · max(0, n_unstaged_rooms − n_covered) · τ              # responsiveness (§7.2)
    − c_move · Σ claimed-carrier busy-seconds this transition        # earn-your-distance (§7.4)
    + γ(τ)·Φ(s′) − Φ(s)                                              # PBRS breadcrumbs
    + B_deliver · (agent-delivered retrieves this transition)
    [ + B_clean at clean-rest, episodic Stage A only ]
```

- **SMDP-correct discounting**: γ(τ) = exp(−τ/T), T ≈ 600 s, applied as a
  per-transition discount vector in GAE (replaces fixed 0.99-per-decision —
  the variable-dt horizon bug). T is long enough that a burial's future dig
  cost is visible at store time; the Φ tidiness term carries the rest.
- **Φ is a function of physical state + task queue only** — never of executor
  commitments (commitment-dependent potentials break telescoping the moment a
  move aborts). Φ = −(Ŵ_deliver + Ŵ_stage + λ_tidy·R_x):
  - `Ŵ_deliver`: per pending retrieve, moves-to-deliver estimate
    (depth + route hops + carry + serve) — O(1) drop per productive move.
  - `Ŵ_stage`: **room-centric** — a function of the room's distance to
    "empty docked here", not of who happens to hold an empty (kills the
    staging/dig conflict for real this time).
  - `R_x`: **convex** retrieval-cost surplus above the layout floor,
    Σ_cars (dig_seconds − floor)², floors calibrated by the oracle — surplus-
    above-floor so clean rest is *exactly* zero (per the recorded calibration
    lesson), convex so the worst-buried SUV dominates. λ_tidy ≈ 0.3.
- **`n_covered`** (the §7.2 coupling, finally well-defined): rooms whose
  serving carrier is currently claimed by any move serving a pending retrieve
  (dig, relay, or delivery leg) — read off the executor registry. This counts
  the *whole* retrieval window, per §7.2's "a room's un-staged window *is* that
  retrieval's in-flight time", not just the final delivery leg.
- **Deleted**: `w_idle`, `w_idle_fixed`, `w_stage`, `fine_scale` annealing,
  `p_deadlock`, `reward_store`, the `RewardNormalizer` (fixed calibrated units
  need no normalization — and per-minibatch advantage normalization must be
  re-examined too, so the rest margin isn't re-buried), the KL leash.
- **At clean rest** every rate is zero, HOLD is free, and a tidying move must
  pre-pay `c_move·duration` against a concrete convex ΔR_x — §5 rest and §7.3
  tidying are the same arithmetic, and voluntary proactive digging is *priced*,
  not forbidden.

Six knobs total: `c_resp, c_move, w_store, λ_tidy, B_deliver, T` — all in
seconds, all eyeball-checkable in the viz.

## 5. Observation & network

Keep the typed-GAT trunk (hidden 96, 2 GAT layers, 7 edge types). Changes:

1. **Two-pointer autoregressive action head** (replaces flat slots): source
   pointer over shelf+carrier node embeddings + a HOLD scalar; destination
   pointer conditioned on the chosen source — `MLP([h_dst; h_src; global])`.
   Joint logp = sum. Size-invariant by construction (scores nodes, not
   positions) — this *is* the campus-transfer bet, strengthened.
2. **Commitments observable** (the verified POMDP hole): per-carrier
   claimed/job features + ETA, per-shelf reservation flags, and an in-flight
   move edge type (src→dst). The policy plans around commitments it can see;
   nobody piles onto a claimed task.
3. **Value head: mean+max concat pooling** + globals (max channel is what sees
   the one worst-buried SUV / the one ignored request that mean-pool averaged
   away). Keep the 7 new global features from the uncommitted diff.
4. From scratch — no warm start (new heads make old checkpoints incompatible,
   and the exploration cliff that made warm-starting necessary is gone).

## 6. Training program

**Stage 0 — infrastructure, learning-free validation.** Build
executor + oracle + masks; drive them with a *random legal-move policy* over
long soaks. Property-fuzz: termination of every script, ledger == sim pending
counts, zero precondition rejections, solvability invariant holds under
concurrent moves, mask never empties while work pending. This de-risks the two
new correctness surfaces before any RL. Also: measure oracle cost per epoch
and move-level decisions/s (expect ≥4× primitive-level experience per
sim-second).

**Stage A — episodic recovery, from scratch.** Terminate at clean rest
(+B_clean). Reset mixture: ~50 % coverage (`shuffle_state`, fullness U(0,0.92),
repair-or-reroll), ~30 % reverse-curriculum tiers T0–T7 re-hosted at move level
(the apex tier is now 6–8 decisions deep — inside PPO's practical exploration
radius), ~20 % `InitialStateSampler` axes (big_ratio, big/small_disorder,
big_shelf_fullness — the untrained §8 lever behind the SUV symptom), all
oracle-certified solvable. Frontier weighting (oversample 50–80 % greedy
buckets). Gate: greedy ≥0.999 on every bucket incl. suv-apex, zero loops.

**Stage B — continuous fluency, same objective** (only B_clean drops; the
rates were always running). K≈16 population, permanent mixture: ~50 %
non-terminating streams (Poisson store 0.01–0.02/s, Γ dwell 150 s, admission
gate on, half clean-start half messy), ~30 % Stage-A episodic rehearsal — the
anti-forgetting anchor, kept **forever** (no objective switch ⇒ nothing to
forget ⇒ no leash), ~20 % **adversarial stream**: a request sampler that asks
for the argmax-R_x car (worst-buried, worst-routed), issues 2–3 concurrent
requests, and biases store sizes big exactly when big-shelf air is scarce
(always through the admission gate, so never unsolvable). This operationalizes
§2's "arbitrary/adversarial" instead of asserting it, and makes
keep-retrievable *emergent*: bury something and the adversary makes you pay
the full dig, undiscounted by luck.

**Stage C — contingency: move-level expert iteration.** Only if the apex/multi
tail stalls under pure PPO: a Gumbel-style short search over the *move* graph
(branching ≤ ~150 masked, depth ≤ ~12, deterministic between events — far
cheaper than primitive-level search) as a training-time policy-improvement
operator; distill; deployment stays policy-only, gated on policy-only metrics
with measured policy↔search agreement. This is not BC-from-an-oracle (the
documented dead end): the teacher is the policy's own search on the policy's
own states, the one imitation regime immune to covariate shift — and at move
level the desync failure mode doesn't exist to begin with.

**Stage D — campus.** Train on a topology mixture {tiny_medipol, campus} (per-
topology `GraphCollator` instances — the code's own comment says that's the
only barrier), oracle in incremental/regional mode. The move-level bet: what
scales with facility size (route length, carrier count, rendezvous scheduling)
is *scripted*; what the net learned (which pallet goes where) is local-
structure reasoning that looks identical on both. Gate zero-shot, then
few-shot, same thresholds.

## 7. Eval gates — all policy-only, all zero-tolerance

Reuse `recovery_eval` / `continuous_eval` / `behavior_eval`, plus:

1. **Recovery**: ≥0.999 greedy on the full battery incl. suv-apex and
   multi-request buckets; steps-to-restore vs oracle bound.
2. **Fluency**: staging_uptime ≥0.95; redundant-move rate ≈0, now *measurable
   exactly* (a move with ΔW≥0 ∧ ΔR_x≥0 is redundant by definition);
   HOLD-fraction at true rest ≈1; latency medians within ~10 % of the measured
   depth floors (45/70/97 s).
3. **Responsiveness (§7.2)**: time-integral of max(0, unstaged − covered) ≈ 0.
4. **Solvability**: zero solvable→unsolvable transitions (structural; audited).
5. **Never-stuck**: zero loops/stalls over ≥10⁶ decision instants of
   adversarial-stream soak; **every** failure oracle-audited (policy gap vs
   mask bug vs unsolvable — the latter two are design failures, not training
   data).
6. **Campus**: zero-shot then few-shot at the same bars.

## 8. Hygiene fixes that stand alone

Worth doing regardless of the redesign (all verified in code):

1. `shuffle_state` repair-or-reroll — never silently accept an unsolvable
   layout (§10 poison, shuffle.py:108-111).
2. `RewardNormalizer(n_envs=1)` across K vec envs — fix or (better) delete.
3. SMDP discounting γ(τ)=exp(−τ/T) in GAE — the fixed-γ-over-variable-dt bug.
4. Committed-destination observability (carrier feature + edge).
5. Convex surplus-above-floor `_r_excess`; room-centric `stage_steps`;
   `c_resp` counting in-flight deliveries, not pending retrieves.
6. Admission gate on by default + in-flight big reservation (concurrency race).
7. Wire the `InitialStateSampler` axes into training resets.
8. Mean+max value pooling.

## 9. Risks, falsifiers, de-risking order

| Risk | Mitigation / falsifier |
|---|---|
| Executor/oracle are new correctness surfaces | Stage 0 property-fuzzing before any RL; sim preconditions as second wall; oracle differential-tested against brute force on small instances |
| Oracle cost per epoch (esp. campus) | memo fast path + exact-on-fail; incremental per size class; measured in Stage 0 with a hard budget — falsified if campus epochs exceed ~50 ms |
| Non-preemptible moves add latency tail | measure vs 45 s depth-0 floor; ABORT-before-TAKE prepared |
| Move-level ping-pong (policy liveness is empirical) | cost bleed + anti-cycle + zero-tolerance soak gate + oracle audit; Stage C teacher if needed |
| Carrier-as-buffer expressiveness loss | eval battery includes stranded-air states; `PARK` move variant prepared |
| Two-pointer head destination fan-out shift on campus (18→145) | topology-mixture training (never sequential transfer); count-normalized features |
| SMDP value-target variance under variable τ | T=600 smoothing; per-env return accounting; validate on Stage A before B |

**Order of work**: executor + reservation ledger → oracle + masks → Stage 0
fuzz + throughput measurement → MacroEnv + reward terms → two-pointer head +
batching → Stage A → Stage B (+ adversary) → gates → campus. The first
falsifiable milestone is cheap: if Stage 0 shows the mask+executor can run a
*random* policy for 10⁶ moves with zero invariant violations and acceptable
throughput, the foundation is sound; if Stage A then fails to crack the apex
tier quickly, Stage C's teacher exists before any architecture change.

## 10. Relationship to the prior campaign

Kept (re-hosted at move level): the coverage doctrine (§8), reverse curriculum,
PBRS discipline, convex R_x concept, the eval-gate posture, the sim, the
typed-GAT trunk, the pointer-head idea, all three evals. Retired: primitive
action space for training, warm-start + KL leash, the fine zoo
(`w_idle`/`w_idle_fixed`/`fine_scale`/`anti_cycle`-as-load-bearing),
`p_deadlock`, `RewardNormalizer`, `reward_store`. Guardrails challenged, with
justification: "no search anywhere" narrows to "no search in the deployed
decision path" (offline certification/auditing and optional training-time
self-distillation are hygiene tools, not crutches); "no post-hoc guards"
already admitted true-physical-rule guards — the solvability mask enforces the
same §10 world-invariant as the admission gate the spec itself assigns to the
environment.
