# Target behavior of the perfect OOS agent

> Status: **living draft**. This captures the *intended behavior and objective* of
> a hypothetical perfect agent. It does not yet prescribe a reward, an env
> subclass, or a training procedure — those come after this is agreed. Terminology:
> `small` = sedan, `big` = SUV (sim naming).

## 1. Scope & training plan

- **Primary target: `tiny_medipol`** (2 lifts L1/L2 serving rooms R1/R2, 2 shuttles
  S1/S2, 16 shelves / 48 slots, 4 handoffs).
- **Then `campus`** — same structural pattern, just bigger (5 lifts + 5 shuttles,
  5 rooms, 140 shelves, 25 handoffs, 420 slots). The bet: the typed-GAT policy
  generalizes from `tiny_medipol` to `campus` with **minimal tuning**, because the
  topology *kind* is identical and only the *count* grows.

## 2. The two unknown distributions — do NOT predict them

Two things are genuinely unknown, and the agent must **not** guess or pre-bet:

1. **Next park size** — whether the next car to park is `small` or `big`. Any type
   can arrive at any time.
2. **Which car is requested, and when** — any stored car can be requested at any
   time.

These are unknown *now* and will only be characterized *after deployment*; later we
may specialize the network to the realized distributions. For training they are
**arbitrary / adversarial**: correctness and robustness must hold for *any*
sequence. The agent earns nothing by anticipating either distribution.

## 3. Concurrency

- **Multi-room parks**: several cars can park at once, using multiple rooms
  simultaneously.
- **Concurrent retrievals**: more than one car can be requested at the same time.

So responses must work when multiple disruptions are live at once, not just one.

## 4. System constraint — minimal free-slot headroom

The facility is **conserved-pallet** and seeded **mostly with empty pallets**. The
scarce resource is therefore **not pallets** but **free slots (air)** — shelf
capacity holding *no pallet at all*. By design there is only **just enough**
free-slot headroom to retrieve any item: enough to relocate blockers and stage
rooms, and no more.

- **Base headroom = total shelf capacity − total pallet count** (fixed, since pallets
  are conserved). This is the maneuvering space available when every pallet sits on a
  shelf. It is small *on purpose*.
- **Free slots are a scarce, shared resource.** A dig relocates each blocker into a
  free slot (or hands it off to a free slot in another region). With minimal headroom
  there is essentially no scratch space — blockers cannot be dumped just anywhere.
- **A careless move can deadlock a solvable state.** Consuming or stranding the last
  useful free slot can make a retrieval impossible even though the state *was*
  solvable. This is the concrete way "never get stuck" (§8) is at risk: the headroom
  is too tight to waste.
- **Big-shelf air is the rarest of all.** Only `big` (SUV) shelves can hold a
  relocated SUV blocker, and they are few. The apex maneuvers in §9 exist precisely
  because big-shelf headroom is minimal — every free big slot must be spent
  deliberately.
- **Keep headroom *usable*, not just available.** Part of "keep the system cheaply
  retrievable" (§7.3) is keeping free slots *reachable by the right carrier* and *of
  the right size class* — air stranded behind the wrong carrier, or on the wrong
  shelf class, is air you cannot use.
- **Saturation makes buried items unretrievable — an admission question.** There are
  enough empty pallets to *fill* the `big` shelves, so it *looks* like SUVs can be
  accepted until those shelves are full. They cannot: once **every** big-shelf slot
  is occupied, a **buried** SUV can never be retrieved — its SUV blockers have nowhere
  to relocate (only big shelves accept them, and no free big slot remains). Refusing
  the one store that would cross this threshold keeps *everything* retrievable. That
  accept/refuse decision is **admission control** — the subject of §10.

## 5. Steady state (rest / equilibrium)

> When there is **no pending retrieval**: every room is **staged** and every
> carrier is **idle (not moving)**.

- "Room staged" = the room's serving carrier is docked at the room holding an
  **empty pallet**, ready for a customer to park onto.
- This is the resting attractor the agent returns the system to after every
  disruption. It is the maximally-responsive configuration.

### 5.1 Rest under empty-pallet scarcity (operator ruling, 2026-07-03)

The §5 rest state bends gracefully as the pool fills:

- **If free empty pallets ≥ rooms → every room staged** (unchanged).
- **If free empties < rooms → stage exactly that many rooms**; the rest
  cannot be staged and are *excused*, not a pathology.
- **At the boundary (no free empty left): when a customer parks onto the
  last staged empty, the car STAYS ON THE LIFT — no storage.** Storing it
  would strand the room un-stageable anyway and waste motion; on the lift
  the car is instantly deliverable when requested. The moment a retrieval
  frees an empty (or one is uncovered), normal store/stage behavior
  resumes.
- A facility in this state with only stores queued is **overload-quiescent**
  — customers wait at the door; it is a legitimate rest, not a wedge.

## 6. The two disruptions and the required response

A customer interaction always **disrupts** equilibrium; the agent's job is to
**restore** it.

### Park (store)

- A customer parks a car onto a room's staged empty pallet → that pallet now holds
  a car → **the room is no longer staged**.
- Required response: **store** that car on a size-compatible shelf somewhere, get a
  **fresh empty pallet** onto the serving carrier, **re-stage** the room, and return
  to idle.

### Retrieve

- A request names a **specific car**. The agent must **bring that exact car to any
  room**. The customer drives it off → the car leaves, the pallet becomes empty →
  **the room auto-stages** → system restored.

## 7. Objective

The agent minimizes a small set of **coupled** costs. They mostly move *together*
(a fast retrieval is also a short un-staged window), with occasional genuine
trade-offs the agent must balance.

1. **Minimize average retrieval duration** (request-arrival → delivery) over all
   requested cars.

2. **Maximize responsiveness = keep rooms staged; un-stage only when a delivery
   requires it.** *(Q1 resolved → Reading A.)*
   - At rest (no pending retrieval), **every** room is staged.
   - A room is un-staged **only when it is crucial for a delivery**: to deliver a
     requested car, the delivery room's serving carrier must free its staging empty
     in order to carry the car, so **that** room transiently un-stages. Rooms not
     needed for any delivery **stay staged**.
   - **The number of simultaneously un-staged rooms should track the number of
     in-flight deliveries — never exceed it.** One live retrieval → at most one room
     un-staged. Two concurrent retrievals → un-staging a second room to speed the
     second delivery is fine. **All** rooms un-staged while only one or two
     retrievals are live is a *pathology* — a signal that something is wrong.
   - **Coupling:** a room's un-staged window *is* that retrieval's in-flight time,
     so minimizing retrieval duration directly minimizes un-staged time. We push
     **both** down — room-unstaged-time and retrieval-time — and accept a deliberate
     trade only when un-staging an extra room genuinely makes a concurrent retrieval
     faster.

3. **Implied dual — keep the system cheaply retrievable** (storage placement).
   Because future requests are unpredictable, *where a parked car is stored* silently
   sets the cost of a future retrieval the agent can't foresee. A perfect agent
   stores cars so that **any** car stays cheap to dig later: don't bury needlessly,
   keep at least one empty pallet reachable, keep scarce `big` (SUV) shelves
   un-polluted, and keep free-slot headroom (§4) usable. *(Q2 resolved → in scope.)*

4. **No redundant motion.** No carrier looping; no repositioning, digging, or any
   work that does not serve objectives 1–3. Every move must earn its distance — a
   carrier with nothing useful to do stays idle.

## 8. Robustness & fluency

Competence must be **fluent** — reliable and automatic across the *entire* problem,
not occasional success up to some difficulty ceiling.

- **No difficulty ceiling.** Solving up to *some* hardness level is **not
  sufficient**. Every case in §9 — including the apex SUV maneuvers — must be
  handled reliably, in whatever combination arises.
- **Always completes.** The agent must **never get stuck, never loop, and never
  fail** to (a) deliver a requested car or (b) re-stage a room. Retrieval and
  acquiring an empty pallet are *skills*, and they are **harder to sustain while the
  system runs continuously** (no convenient episode reset to bail out to).
- **Self-induced order — its own "perfect working."** Pursuing the objective is
  inherently tidying: under its own policy the realized state gets progressively
  cleaner and more retrievable, converging toward a low-entropy, maximally-responsive
  configuration. This is the state distribution the agent *creates for itself*.
- **Full-state-space fluency — do NOT overfit to that self-induced distribution.**
  The agent must complete any task from **any arbitrary (solvable) state**, including
  messy, high-entropy states it would *never itself produce*. This is the central
  trap: a policy that is good *because* it keeps things tidy only ever sees tidy
  states, never practices hard recoveries, and so **breaks when handed a disordered
  state** at deployment (or after any disturbance). Training coverage must therefore
  span the **whole** state / hardness space — arbitrary fullness, big-shelf
  saturation, blocker depth, stack disorder — *independently* of what the policy
  would naturally visit. (The `InitialStateSampler` / `shuffle_state` knobs —
  `fullness`, `big_ratio`, disorder, `require_solvable` — are the lever for this.)

## 9. Hardness taxonomy (difficulty axes / curriculum)

Listed simplest → hardest; **incomplete** (examples to anchor the difficulty
structure, not an exhaustive enumeration).

1. **Direct retrieve, depth-0, slot available.** Requested car is on top (depth 0)
   of a shelf the serving carrier reaches, AND a free slot exists on a shelf the
   serving carrier reaches. → Serving carrier offloads its staging empty onto the
   free reachable slot, goes to the car's shelf, TAKEs it, delivers to the room.
2. **Direct retrieve, depth-0, no reachable free slot.** As above but the serving
   carrier has nowhere of its own to put the empty pallet → it must **hand off** the
   empty to another carrier to free itself, then take the requested car.

Difficulty then scales along these axes (composably):

- **Depth `d > 0`** of the requested car → its `d` blockers above it must be
  relocated first.
- **Handoff required** for the empty pallet and/or for blockers, and **how many**.
  Worst case: no reachable free slot at all ⇒ the empty pallet **and every blocker**
  must be routed out via handoff (nothing can be placed directly).
- **`big` (SUV) shelf interactions (apex difficulty).** SUV shelves are **scarce**;
  they may be full of SUVs, or **polluted** with non-SUV items, which constrains
  where SUVs can be shuffled and forces complex maneuvers. Hardest known case:
  **temporarily relocate an SUV onto the requested car's SUV shelf** in order to
  free a non-SUV blocker off *another* SUV shelf, so the requested car's blockers
  can be fully cleared — possibly requiring handoffs as well.

## 10. Solvability invariant & training-signal hygiene

A reached or randomly-generated state can be **unsolvable**: a requested car whose
retrieval has *no legal solution* (§4 — its blockers cannot be relocated anywhere).
That is not only an operational failure; it is a **training hazard**.

- **Always solvable ≠ always easy.** We forbid only the *no-solution* case. A
  retrieval that is deep, multi-handoff, or needs the apex SUV maneuver (§9) is
  expensive but solvable — exactly what we *want* to train on. The line we draw is
  solvable-vs-unsolvable, never easy-vs-hard.
- **The invariant: the world is always solvable** — every stored car retrievable at
  all times, so every Retrieve that can be issued has a solution. Held on three
  fronts:
  1. **Solvable initialization** — episodes start solvable only (the `require_solvable`
     re-roll / repair in `shuffle_state` / `InitialStateSampler`).
  2. **Admission control** — the system refuses any store (notably an SUV) that would
     make the layout unsolvable, so over-acceptance can't create a dead state. The sim
     already has this (`gate_big_retrievability` / `_big_admission_ok`), currently
     **off**; turning it on (and generalizing it) is the lever.
  3. **The agent never breaks solvability** — through its own relocations it must never
     strand the headroom a future retrieval needs (preserve-solvability, §8).
- **Why it poisons RL.** If a Retrieve is ever issued for an unretrievable car, the
  agent *cannot* complete it; a completion-based reward then emits a persistent,
  un-actionable signal — the agent learns noise, or degenerate give-up / thrash. So
  the agent must **never be trained against an unsatisfiable task**: no unsolvable
  starts, and no request for a car that isn't currently retrievable.

## 11. Open questions (to resolve before reward design)

- **Q1 — RESOLVED → Reading A.** Keep the most rooms staged; un-stage a room only
  when crucial for a delivery; un-staged-room count tracks in-flight deliveries.
  See §7.2.
- **Q2 — RESOLVED → in scope.** *Where* a parked car is stored is part of "perfect":
  the agent is judged on keeping the system cheaply retrievable for unknown future
  requests, not only on executing each retrieval once requested. See §7.3.
- **Q3 — admission control: environment guarantee or agent decision?**
  *Recommendation: environment.* The agent can't see the next car's size (§2) and has
  no reject primitive (a customer parks onto any staged empty), so type-selective
  admission isn't agent-doable — the sim should drop solvability-breaking stores at
  arrival, and the agent then only ever faces solvable worlds. Confirm, or do you want
  the agent to bear part of it (e.g. by choosing not to stage a room)?
