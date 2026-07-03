# The move-level stack — implementation notes

> Companion to [SOLUTION_V2.md](SOLUTION_V2.md) (the design). This documents
> what was actually built, where it deviates from the design and why, and how
> to run it. Updated as of Stage A/B training on `tiny_medipol`.

## Module map

| Module | Role |
|---|---|
| `oos/plan/oracle.py` | `SolvabilityOracle` — exact solvability for move semantics: closed-form counting (aggregate pass + O(depth)/car) + extraction closure for creatable big air; `audit_retrievable` (bounded DFS with airborne holds) kept offline for differential testing. `admission_ok` = the environment-side gate. |
| `oos/env/moves.py` | `Move`, `MoveExecutor` — enumeration of startable moves (physical legality + shelf locks + free-chain + oracle mask), atomic all-free carrier claims, per-carrier primitive scripts with passive rendezvous steps, analytic makespan/busy estimates, immediate-inverse bookkeeping. |
| `oos/env/move_env.py` | `MoveEnv` — the semi-MDP: pump loop (event-exact piecewise cost integrals), reward, Φ, resets (coverage / sampler-axes / tiers / clean / continuous), `GatedEngine` (oracle admission for **all** store sizes), adversarial request stream, anti-cycle penalty, immediate-inverse guard. |
| `oos/learn/move_net.py` | Two-pointer `MovePolicyNet` (typed-GAT trunk reused; src pointer + HOLD, conditioned dst pointer, mean+max value pooling), `MoveCollator`. |
| `oos/learn/move_ppo.py` | Rollout + PPO with composite (src,dst) log-probs and per-transition SMDP discount γ_t = exp(−τ_t/T) in GAE. No reward normalizer. |
| `oos/learn/train_move.py` | Stage A/B trainer (`python -m oos.learn.train_move --stage a` / `--stage b`). |
| `oos/learn/move_eval.py` | Greedy recovery battery (per-tier buckets) + continuous soak (staging uptime, latency, HOLD-at-rest, §7.2 excess integral, redundant-tidy, W-progress stuck watchdog). |
| `oos/learn/move_certify.py` | Certification suite: n≥200/bucket battery + soaks + never-stuck run + per-failure audit (loop vs timeout; final-state solvability). |
| `oos/learn/move_deploy.py` | Policy-only continuous deployment CLI. |
| `tests/test_move_fuzz.py` | Stage-0 property fuzz: invariants under a random policy across all env modes. |

Everything under `oos/sim/` is untouched; the primitive-level env/learn stack
remains for the viz.

## Deviations from SOLUTION_V2, and why

1. **The oracle needs no runtime search.** The design specified exact-search-
   with-memo-fast-path. Working the algebra of pop→push semantics gave a
   closed form instead: shelf→shelf moves conserve total air and nothing
   re-enters below a target, so retrievability ≡ (blockers fit in off-shelf
   air) ∧ (big blockers fit in big air **after the extraction closure**,
   where extractions convert small-air to big-air 1:1 and displaced bigs may
   temporarily use the target shelf's own air — the §9 apex play — plus lead
   non-big blockers may depart first to grow that hop space). Differential-
   validated at **100% agreement over 6,655 random states on three
   facilities** against exhaustive move-space search, both directions.
2. **`max_holds=1` is the correct abstraction, by construction.** k
   concurrent moves ≡ k sequential pop→push events for reachability, so the
   oracle's solvability notion is exactly "solvable via moves". Multi-hold
   (carrier-as-buffer) plans are excluded from the *reachable set* by
   admission control — consistent with §4's "admission keeps everything
   retrievable". The audit search's goal explicitly requires all holds
   landed, or it would silently model the richer buffer semantics.
3. **Admission gates ALL sizes, on the future view.** The design's in-flight
   big reservation generalizes: pending stores are projected into the future
   view as held cars, so the agent's own move mask can never strand the
   headroom an admitted-but-unparked store needs, and near-simultaneous
   stores cannot jointly overrun (the race the review found).
4. **A family of structural guards accrued during training, each a
   provably-wasted-work exclusion (not a skill substitute), each with the
   stall-safety escape (a guard may never empty the productive mask):**
   - *self-staging no-op*: `MOVE(carrier→room)` when already docked there
     compiles to an empty script — a zero-time free action greedy argmax
     got stuck on; masked.
   - *immediate inverse*: the pallet of the single most recently completed
     move may not go straight back where it came from while nothing else
     changed. The apex return is unaffected (the extraction in between
     clears the block).
   - *config cycle guard* (sequential epochs only, `n_inflight == 0`): a
     move whose predicted post-configuration — shelf arrangement + room
     staging disposition — was already seen since the last task-set change
     is masked. Kills k-cycles the 1-deep inverse guard misses, including
     the two-rooms-swap-one-empty loop; restricted to the sequential regime
     because airborne pallets make the shelf picture underdetermine the
     state under concurrency (a first, unrestricted version measurably hurt
     the concurrent tiers).
   - *§5 rest-attractor rule*: HOLD (literal idleness) is not offered while
     an unexcused un-staged room has a legal SINGLE-CARRIER staging move
     (its own serving carrier fetching a reachable empty). The policy still
     chooses which productive move; relay-chain staging is never forced
     (forcing it contended with in-flight digs — t7 regression evidence).
5. **Anti-cycle penalty** (0.5 on physical-state revisit while work pending,
   visited-set cleared on any task-set change): at move level a revisit
   provably wasted work, so the penalty cannot fight a legitimate maneuver.
   Added after failure forensics showed greedy loops were 11/13 of residual
   battery failures; it moved the first post-fix battery to 0.991.
6. **PBRS uses the undiscounted Φ′−Φ form** even though returns are
   discounted: with Φ ≤ 0, γ(τ)·Φ′−Φ pays a positive drip for idling in bad
   states (the pathology `reward_gamma=1` fixed in the primitive stack).
7. **Reset hygiene went beyond the design list**: repair-or-reroll verifies
   with the MOVE oracle (stronger than `_layout_is_solvable`); parked cars
   placed by coverage resets are verified against the full view (a big car
   swaps an empty for a car in the accounting — reverted if it breaks
   solvability); `InitialStateSampler` axes are wired as a 35% coverage
   flavor with oracle repair on top.

## Stage-0 results (the falsifiable milestone)

- ~40k random-policy decisions across episodic / continuous / adversarial /
  apex-tier modes: **0 stalls, 0 invariant violations, 0 executor errors**.
- Throughput ≈ 2,200 env decisions/s (vs ~250 primitive steps/s in the old
  stack), sim speedup ≈ 6,000×.
- Oracle differential: 100% (see above), 0 budget exhaustions on the hot
  path (there is no search on the hot path).

## Training

Stage A (episodic recovery, from scratch, 282k params): the first battery at
iteration 20 already reached 0.72 greedy success (the primitive-level
from-scratch baseline was 0.000 after 22 iterations); with the anti-cycle
term and the inverse guard the battery runs in the high 0.9s and the
remaining failures are audited per-episode (every failure so far: policy
gap; **zero** ended in an unsolvable state — the invariant holds under fire).

Commands:

```bash
python -m oos.learn.train_move --stage a --iters 400 --out runs/move
python -m oos.learn.train_move --stage b --iters 400 --out runs/move \
    --resume runs/move/stageA_best.pt
python -m oos.learn.move_certify --ckpt runs/move/stageB_best.pt \
    --n-per-bucket 200 --out runs/move/certify.json
python -m oos.learn.move_deploy --ckpt runs/move/stageB_best.pt \
    --sim-hours 8 --adversarial
```

## Certified results (v1: `runs/move/stageB_final.pt`, 282k params)

Full certification (`runs/move/certifyB_final.json`), all policy-only greedy:

- **Battery, n=200/bucket (1,800 episodes):** 0.9817 overall, min bucket
  0.9250 (t7-full). **0 loops** (the guard family removed the entire loop
  failure class), **0 unsolvable-after** — every failure is a timeout on a
  hard concurrent state; the solvability invariant never broke anywhere.
- **Never-stuck:** 6 sim-hours of maximum-hostility adversarial stream
  (worst-buried-car requests + big-biased stores): 556/561 delivered,
  **0 stalls, 0 stuck flags, 0 wedges** over 6,704 decisions.
- **Fluency:** normal-stream §7.2 excess-unstaged 92 s/h (gate ≤150),
  HOLD-at-rest 1.000, redundant tidy moves 0. Latency 71.5 s mean /
  144.6 s p95 (normal); adversarial excess is sample-noisy (152–625 s/h
  across evals — back-to-back worst-case digs dominate).

Open gaps → the design's designated LAST lever, capacity (hidden 128 /
3 GAT layers, fresh pipeline on the final ruleset): the t7 tail and
adversarial responsiveness variance.

## SHIPPED: `runs/move_big/stageB_final.pt` (618k params, capacity pipeline)

Fresh Stage A → Stage B on the final ruleset; certification in
`runs/move_big/certify_final.json`:

- **Battery, n=200/bucket:** 0.9783 overall, min bucket 0.900 (t7);
  **0 loops, 0 unsolvable-after** — all 39 failures are timeouts on hard
  concurrent states.
- **Fluency — every gate passes, including adversarial:** §7.2 excess
  131.7 s/h normal / **133.2 s/h adversarial** (v1 was 625) / **61.4 s/h**
  over the 6-hour never-stuck run; HOLD-at-rest 1.000; 0 redundant moves.
- **Never-stuck:** 6 adversarial sim-hours, 543/546 delivered (rest pending
  at cutoff), **0 stalls / 0 stuck / 0 wedges** over 7,727 decisions;
  adversarial soak delivered 170/171.

v1 (`runs/move/stageB_final.pt`, 282k) is kept as an alternate: marginally
better battery tail (0.925 vs 0.900 min bucket), much worse adversarial
responsiveness. The capacity model is the deployment recommendation — it is
the better *continuous* agent, which is the product.

## Known limitations / next steps

- **Campus**: the stack is size-invariant by construction (pointer heads,
  per-node features, per-topology collator), but campus training (Stage D)
  has not been run yet; the oracle's aggregate pass is O(shelves) per check
  and will want the incremental/regional variant there.
- **Viz**: the DearPyGui viz drives the primitive `Environment`; a MoveEnv
  playback shim is future work (the sim state is shared, so a thin adapter
  that replays executor primitives through `Session` is straightforward).
- **Carrier-as-buffer moves** (`PARK` at a pose) remain unimplemented by
  design; admission control keeps the system inside the moves-solvable
  region, at a small measured admission-refusal cost.
- **Admission is deliberately conservative for big stores in tight states**:
  the hands-aware check must hold for EVERY serving carrier that might
  absorb the store (the sim's auto-serve picks whichever room stages first,
  which is unknowable at queue time). Certified drop rates: ~5–19 stores
  per multi-hour window under heavy streams; the alternative is the
  recorded both-lifts-wedged deadlock.
