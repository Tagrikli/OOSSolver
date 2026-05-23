# The OOS planner problem

## Brief summary

A Parkolay facility is an automated storage/retrieval system. Customers arrive
at rooms to drop off items (store) or pick up items (retrieve). Items always
sit on pallets; pallets move through the facility on a network of carriers
(shuttles move horizontally, lifts move vertically) that hand off to each
other and to LIFO shelves. The planner decides, continuously and concurrently
across all carriers, how to service an incoming stream of store and retrieve
tasks so that the **expected per-task wait time** is minimized — where store
cost is "customer arrival → room available with an empty pallet" and retrieve
cost is "request → item delivered to a room." Between tasks the system may
reshuffle pallets to improve future expected cost, but never at the price of
leaving rooms unstaged.

## Theoretical formalization

### Static facility structure

A facility is a tuple `F = (C, S, R, P, A, T, H, κ, σ, ρ)`.

- `C` — finite set of **carriers**. Each `c ∈ C` has a type
  `τ(c) ∈ {shuttle, lift}` and a finite set of accessible positions
  `Pos(c)`. Carriers hold at most one pallet, regardless of contents.
- `S` — finite set of **shelves**. Each shelf `s ∈ S` has:
  - capacity `κ(s) ∈ ℕ₊`,
  - size class `σ(s) ∈ {small, big}`,
  - access set `A(s) ⊆ C`. For non-transfer shelves `|A(s)| = 1`; for
    transfer shelves `|A(s)| = 2`, and a width flag
    `w(s) ∈ {narrow, wide}` further refines coupling (see dynamics).
- `R` — finite set of **rooms**. Each room `r ∈ R` is served by exactly
  one carrier `ρ(r) ∈ C`.
- `P` — finite set of **pallets**. Pallets are fungible (interchangeable
  identities).
- `H` — finite set of **handoff poses**. Each `h ∈ H` is a pair
  `(c_a, c_b)` of carriers that can exchange a pallet at a designated
  position `pos(h, c_a) ∈ Pos(c_a)`, `pos(h, c_b) ∈ Pos(c_b)`.
- `T` — the subset of `S` that are transfer shelves
  (`T = { s ∈ S : |A(s)| = 2 }`).

Define the **item universe** `I` (unbounded), with size `size(i) ∈ {small, big}`
for each item `i ∈ I`. An item is **size-compatible** with a shelf `s` iff
`size(i) = small ∨ σ(s) = big`.

### Dynamic state

The state at continuous time `t ≥ 0` is `x(t) = (X_S, X_C, X_R, Q)` where:

- `X_S(s, t)` is the **stack content** of shelf `s`: a finite LIFO sequence
  `(p_1, …, p_k)` with `k ≤ κ(s)`, where each `p_j` is either `(pallet, ⊥)`
  (empty pallet) or `(pallet, i)` for some item `i ∈ I`. The top of the
  stack is `p_k` (the most recently `give`n).
- `X_C(c, t) = (pos_c, load_c)` — carrier `c`'s current position
  `pos_c ∈ Pos(c)` and current load `load_c ∈ {⊥} ∪ ({pallet} × (I ∪ {⊥}))`.
- `X_R(r, t)` — what currently sits at room `r`: either `⊥`, an empty
  pallet (i.e., the serving carrier is at the room with an empty pallet),
  or `(pallet, i)` mid-customer-interaction.
- `Q(t)` — the queue of **active tasks** known to the system, each of form
  `Store(arrived_at, size, room_or_⊥)` or `Retrieve(arrived_at, item)`.

### Initial conditions

At `t = 0`: `Q = ∅`, every shelf holds zero or more empty pallets (the
pre-seeded distribution), every carrier is at some default position with
`load = ⊥`. The total pallet count is conserved for the lifetime of the
facility: items enter and leave through rooms, but pallets do not.

### Action set

An **action** is one of the following primitives, parameterized by the
carrier(s) it acts on:

1. `move(c, pos)` — carrier `c` moves from its current position to
   `pos ∈ Pos(c)`. Duration `d_move(c, pos_c, pos)`.
2. `give(c, s)` with `c ∈ A(s)` — `c` pushes its loaded pallet onto the
   top of shelf `s`. Precondition: `load_c ≠ ⊥`, `pos_c` is the shelf
   position of `s` for `c`, `|X_S(s)| < κ(s)`, and (if the pallet carries
   an item) the item is size-compatible with `s`.
3. `take(c, s)` with `c ∈ A(s)` — `c` pops the top of shelf `s` onto
   itself. Precondition: `load_c = ⊥`, `pos_c` is the shelf position,
   `|X_S(s)| ≥ 1`.
4. `handoff(c_a → c_b, h)` for some `h ∈ H ∪ T_narrow`. Precondition:
   `pos_{c_a} = pos(h, c_a)`, `pos_{c_b} = pos(h, c_b)` simultaneously,
   `load_{c_a} ≠ ⊥`, `load_{c_b} = ⊥`. Effect: instantaneous transfer of
   the pallet from `c_a` to `c_b`.
5. `room_stage(c, r)` — synonym for `move(c, pos_room(r))` while
   `load_c` is an empty pallet, exposing the pallet for customer load.
6. `customer_load(r, i)` — exogenous; transitions room `r` from empty
   pallet to `(pallet, i)`. Latency: customer-controlled.
7. `customer_unload(r)` — exogenous; transitions room `r` from `(pallet, i)`
   to empty pallet (item leaves the facility).

A **plan** at time `t` is a per-carrier sequence of primitives, with
synchronization constraints at handoff/narrow-transfer events that bind
two carriers' schedules at a common instant.

### Concurrency model

Carriers' motion regions are pairwise disjoint. Therefore the only
binding constraints across carriers are:

- **Handoff poses & narrow transfer shelves.** Both involved carriers
  must be at the matching positions *at the same instant*.
- **Wide transfer shelves.** Functions as a one-slot (or capacity-bounded)
  LIFO buffer shared between two carriers; `give` and `take` need not be
  simultaneous, but standard shelf semantics apply.

Hence a per-task plan is a set of carrier-local action sequences plus a
partial order of synchronization events.

### Tasks and cost

The exogenous task stream is a marked point process
`{(t_n, θ_n)}_{n ≥ 1}` on `ℝ₊` where each `θ_n` is either:

- `Store(size_n)`, requiring the planner to pick a room and produce a
  ready-state for that room, or
- `Retrieve(item_n)`, requiring the pallet carrying `item_n` to be
  brought to some room.

The **per-task cost** is:

```
cost(Store_n)    = t_room_available(n) - t_n
cost(Retrieve_n) = t_completed(n)      - t_n
```

where `t_room_available(n)` is the first instant at which the assigned
room holds an empty pallet with its serving carrier present, and
`t_completed(n)` is the first instant at which the requested item sits at
some room ready for `customer_unload`.

### Distribution and the objective

The joint law of `{(t_n, θ_n)}` and of the per-retrieve item identities
is **unknown at deployment** but is approximated by an estimator
`D̂_t` maintained online from observed traffic. Let `π` denote the
planner's policy: a (possibly stochastic) map from state, queue, and
estimator `D̂_t` to action sequences.

The objective is the long-run expected per-task cost:

```
J(π) = lim sup_{N→∞}  (1/N)  E_{stream ~ D, π} [ Σ_{n=1}^N cost(θ_n) ]
```

The planner seeks `π* ∈ arg min J(π)`. Because `D` is unknown,
operationally the planner targets `arg min_π J_{D̂_t}(π)` with `D̂_t`
refining over time.

### Hard responsiveness constraint

Let `r ∈ R` be a room and let `t_next_store(r)` denote the planner's
estimate of the next store arrival at `r` under `D̂_t`. The **room-staged
predicate**:

```
ready(r, t) := X_R(r, t) is an empty pallet  ∧  pos_{ρ(r)}(t) = pos_room(r)
```

The hard constraint on any background work is:

```
∀ r ∈ R, ∀ t :  τ_ready(r, t) ≤ t_next_store(r) - t
```

where `τ_ready(r, t)` is the time needed under the current commitments to
return room `r` to `ready(r, ·)`. In words: background work is admissible
only if it leaves enough slack for every room to re-stage before its next
expected customer.

### Decision variables under π

A policy must, at any time, specify:

1. **Room choice** `μ_room(Store_n; x, D̂_t) → R` — only meaningful when
   multiple rooms accept the size.
2. **Landing shelf** for newly stored items
   `μ_land(item; x, D̂_t) → S`, subject to size compatibility and
   capacity.
3. **Eviction destinations** for blockers during retrieves
   `μ_evict(p_blocker; x, D̂_t) → S`.
4. **Task scheduling order** — a permutation of the active queue `Q(t)`
   (not necessarily the arrival order; reordering is admissible iff it
   reduces the *sum* of per-task costs).
5. **Per-carrier action sequencing** consistent with the synchronization
   semantics above.
6. **Background reshuffling** — an additional, fully preemptable plan
   subject to the hard responsiveness constraint.

### Complexity

Even the deterministic single-task subproblem (one retrieve under known
state, plan the carrier sequence including blocker evictions to minimize
its makespan) is at least as hard as classical block-relocation /
container pre-marshalling, which are NP-hard. The full problem stacks
multi-agent concurrency with synchronization, online distribution
estimation, and policy choice on top of that. Practical solutions
decompose into the four-layer pipeline described in the detailed
explanation below.

## Detailed explanation

### Facility topology

A facility is a DAG of carriers. Two kinds of carriers exist for this
problem (turntables are treated as transparent utility and ignored):

- **Shuttle** — horizontal motion along its track. Carries at most one pallet.
- **Lift** — vertical motion along its column. Carries at most one pallet.

Each carrier has a fixed set of *accessible positions*. Some accessible
positions are *shelf positions* (where the carrier can `give` to or `take`
from a shelf). Others are *handoff positions* (where the carrier can
exchange a pallet with another carrier). A position that is a shelf
position for carrier A is, in general, not a shelf position for carrier B —
shelves are single-owner except for transfer shelves (see below).

Two carriers' motion spaces never overlap. Apart from the exact moments they
meet at a handoff, carriers can act fully independently. This is the
property that makes the problem genuinely concurrent.

#### Transfer shelves and handoff poses

The DAG edges between carriers are of two kinds:

- **Handoff pose.** A pose on carrier A's track that coincides with a pose
  on carrier B's track. To exchange a pallet, both carriers must be
  simultaneously parked at the pose. Nothing is stored there.
- **Transfer shelf.** A real shelf that is accessible to *two* carriers.
  Width matters:
  - *Narrow transfer shelf*: degenerate — it behaves like a handoff pose;
    both carriers must be present to perform the exchange.
  - *Wide transfer shelf*: can hold the pallet temporarily. Carrier A may
    `give` a pallet, leave, and carrier B may `take` from it later. This
    decouples the handoff in time and is a real degree of freedom for the
    scheduler.

Shelves that are not transfer shelves belong to exactly one carrier.

### Shelves

A shelf is a LIFO stack with finite capacity. Indexing convention: the slot
last `give`n into is the slot the next `take` will remove.

Shelf size class:
- *Small shelf*: accepts an empty pallet or a pallet carrying a small item.
- *Big shelf*: accepts an empty pallet, a pallet with a small item, or a
  pallet with a big item.

A carrier's `give(shelf)` operation pushes the pallet it holds onto that
shelf's top, provided (i) size compatibility, (ii) the shelf has free
capacity. A carrier's `take(shelf)` pops the top pallet onto the carrier,
provided the carrier is empty and the shelf is non-empty.

### Pallets and items

- Pallets are fungible. The system never cares *which* empty pallet is at a
  location, only that one is.
- Items have an identity. For a *store*, the identity is irrelevant — the
  planner only needs to know the size class to pick legal shelves. For a
  *retrieve*, the identity is the entire point: the specific item, on its
  specific pallet, currently at a specific depth in a specific shelf, is
  the target.
- An item is moved by moving the pallet under it. An item is never directly
  manipulated.

### Rooms

A room is a special point in the facility:
- reachable by exactly one carrier,
- where a customer interacts with the system: they load an item onto the
  pallet sitting at the room (store) or unload an item from the pallet
  sitting at the room (retrieve).

A room is *ready to accept a store* when its serving carrier is parked at
the room with an empty pallet on it. Anything else is "not ready" from the
customer's perspective and accumulates cost (see below).

### Tasks

#### Store

Customer arrives at time `t_arrival`. They need to drop an item of known
size class.

1. Planner directs them to a room (or, if multiple rooms are valid, picks one).
2. Planner ensures (or has already ensured) that the chosen room is ready:
   serving carrier present, empty pallet on it.
3. Customer loads the item at time `t_room_available ≥ t_arrival`.
4. Behind the scenes, the loaded pallet is then moved to a compatible
   shelf somewhere in the facility. This phase has no customer-visible
   cost — but it occupies the carrier, blocking other use of it.

**Store cost = `t_room_available - t_arrival`.**

#### Retrieve

Request arrives at time `t_request` naming a specific stored item.

1. Planner locates the item: it sits at depth `d` on shelf `s` reachable by
   carrier `c`.
2. The `d` pallets above it (blockers) must be evicted to other valid
   shelves. The planner chooses each eviction destination subject to size
   compatibility, capacity, and overall cost — including potentially
   "anywhere is fine right now, sort it out later" semantics, because
   evicted pallets can be moved again.
3. The target pallet is taken and routed (possibly across multiple carriers
   via handoffs / transfer shelves) to *any* room, where the customer
   unloads the item.

**Retrieve cost = `t_completed - t_request`.**

### Concurrency and scheduling

Carriers act in parallel; the only inter-carrier coupling is at
handoffs / transfer shelves. The planner is multi-agent: a single ongoing
task does not freeze the rest of the facility.

The scheduler may reorder pending tasks. The decision rule is *total-cost
based*, not first-come-first-served: if delivering retrieve B before A
reduces the sum of waits across all pending tasks, the planner does so,
even if A waits longer than it otherwise would. This rules out greedy FIFO.

### Background reshuffling

When no task is active or when carriers are idle within an active task,
the planner may reshuffle pallets — move them between shelves, change which
empty pallets sit where, reduce burial of items it estimates are likely to
be requested soon, etc. Two hard rules:

1. **Room staging dominates.** No reshuffle may leave any room
   unstaged at a moment where a store could plausibly arrive. Concretely,
   the carrier serving a room is either at the room with an empty pallet,
   or able to reach the room within an acceptable window. The planner's
   model of arrival rate informs what "acceptable" means.
2. **Preemptable.** Background work must be designed to abort cheaply.
   When a task arrives, the reshuffle yields, leaving the facility in a
   valid (if non-ideal) state.

### Objective

The system runs for a long time as a continuous stream of tasks. The
quantity to minimize is the expected per-task cost in steady state:

```
E[cost] = E[store_cost over store tasks] + E[retrieve_cost over retrieve tasks]
```

This is response time, not makespan and not summed carrier-time. A move
that ties up a carrier for 60 seconds is fine if it doesn't delay any
customer; the same move is bad if a customer is waiting on that carrier.

The task distribution — store/retrieve mix, size mix on stores, identity
distribution on retrieves (including any dwell-time effect where recently
stored items are more or less likely to be requested) — is **unknown at
deploy time but estimated online**. The policy therefore consumes a
*current estimate* of the distribution, and benefits from that estimate
improving over time.

### What makes the problem hard

1. **LIFO with blockers.** Direct routing is straightforward; routing
   through K-deep burial creates a combinatorial choice (where do each of
   the K evicted pallets go?) that interacts with future tasks.
2. **Multi-agent concurrency with handoffs.** Two carriers can work in
   parallel, but a transfer shelf or handoff pose is a synchronization
   point that constrains their schedules.
3. **Stochastic objective with online learning.** "Best place to put this
   pallet" depends on what will be requested next. That depends on a
   distribution we don't know exactly and that may itself drift.
4. **Foreground/background tension.** Reshuffling reduces future cost
   but costs carrier-time now and risks stranding rooms.
5. **Scheduling vs. routing entanglement.** Reordering pending tasks
   changes which carriers are busy when, which changes which routes are
   feasible, which changes which order minimizes cost. The scheduling
   layer and the routing layer don't cleanly separate.

### Decomposition the planner naturally takes

1. **Online distribution estimator.** Maintains current beliefs about:
   store arrival rate per room, store size mix, retrieve arrival rate,
   per-item retrieval-likelihood (possibly conditioned on time-since-store).
2. **Per-task concurrent planner.** Given current state + estimator output,
   plans the carrier moves to service one named task as fast as possible,
   choosing eviction destinations and handoff timings. This is the
   routing-with-blockers piece — CP-SAT-shaped in the deterministic single
   task limit.
3. **Storage policy.** At store time, picks the room (when there's choice)
   and the eventual landing shelf. The criterion is expected future
   retrieval cost of this item plus expected unblocking cost it imposes on
   pallets beneath it.
4. **Background reshuffle policy.** Decides what idle work to do.
   Foreground constraint: never strand a room. Foreground priority: stage
   rooms with empty pallets. Secondary: de-bury likely-requested items,
   defragment loaded pallets out of small shelves that should hold empties,
   rebalance empties toward rooms.

These four pieces share state and must not contradict each other; an
integrated formulation (e.g., one MDP / one learned policy over the full
state) is the long-term direction, with the explicit decomposition above
serving as both a baseline and a debugging surface.
