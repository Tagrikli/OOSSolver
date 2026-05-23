# Solution 1 — GNN encoder + pointer-attention policy + PPO

Reinforcement-learning agent that controls the facility end-to-end. The
agent observes the facility as a typed graph, encodes it with a graph
neural network, picks one carrier action per decision instant via
pointer attention over node embeddings, and is trained against the
`parkolay-sim-engine` simulator with PPO. Designed so a single trained
agent generalizes across facility configurations (variable carrier /
shelf / room counts and connectivity) and adapts online to the live
task distribution after deployment.

## How the pieces compose

```
                     ┌─────────────────────────────────────┐
                     │           parkolay-sim-engine        │
                     │  deterministic facility physics +    │
                     │  exogenous task stream sampler       │
                     └────────────────┬─────────────────────┘
                                      │
                            event-driven semi-MDP
                          (clock advances to next event;
                          policy queried at decision instants)
                                      │
                                      ▼
                     ┌─────────────────────────────────────┐
                     │       observation builder            │
                     │  current state → typed graph         │
                     │  + flag the querying carrier         │
                     │  + recompute legal-action mask       │
                     └────────────────┬─────────────────────┘
                                      │
                                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │                    policy / value network                     │
   │                                                              │
   │   ┌─────────────────────────────────────────────────────┐    │
   │   │ heterogeneous projection                            │    │
   │   │   carrier / shelf / room features →  ℝ^64           │    │
   │   └────────────────────────┬────────────────────────────┘    │
   │                            ▼                                 │
   │   ┌─────────────────────────────────────────────────────┐    │
   │   │ K rounds of graph attention (GAT)                   │    │
   │   │   each node attends over its typed neighbors        │    │
   │   │   committed-target edges carry intent forward       │    │
   │   └────────────────────────┬────────────────────────────┘    │
   │                            ▼                                 │
   │              per-node embeddings  h_v  ∈ ℝ^64                │
   │                            │                                 │
   │           ┌────────────────┴────────────────┐                │
   │           ▼                                 ▼                │
   │   ┌──────────────┐                ┌────────────────────┐     │
   │   │ value head   │                │ pointer-attention  │     │
   │   │ pool + MLP   │                │ action head        │     │
   │   │ → V(state)   │                │ score(h_c, h_v)    │     │
   │   └──────────────┘                │ mask illegal       │     │
   │                                   │ softmax            │     │
   │                                   │ → (type, target)   │     │
   │                                   └────────────────────┘     │
   └──────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                              env.step(action)
                                      │
                                      ▼
                          reward, next state, done
                                      │
                                      ▼
                          PPO trainer collects, computes
                          advantages, clipped surrogate update
```

## 1. Environment interface

Event-driven semi-MDP wrapping `parkolay-sim-engine`.

- `reset(seed, config_sampler, distribution_sampler)`:
  - Samples a facility configuration (topology, capacities, pallet
    pre-seeding) from the configured distribution.
  - Samples a task stream (store arrival rate, retrieve arrival rate,
    item-size mix, retrieve-likelihood model).
  - Advances simulated time to the first decision instant.
  - Returns the initial observation.
- `step(action)`:
  - Applies the chosen primitive (`take` / `give` / `handoff` / `park`)
    on the querying carrier.
  - Advances simulated time to the next decision instant (next carrier
    that needs orders, next task arrival, next move completion).
  - Computes the reward accrued over the elapsed interval.
  - Returns `(observation, reward, done, info)`.

A **decision instant** is any simulator clock value where at least one
carrier needs an order. Multiple carriers can need orders at the same
instant; the env queries the policy once per carrier with the clock
frozen, then advances. Each subsequent query sees the previously
committed intents (via the `committed_target` edge added to the graph
in observation rebuilds).

## 2. Observation = typed graph

### Node types and features

**Carrier nodes.** One per carrier in the facility.
- `kind`           — shuttle / lift (one-hot)
- `position`       — normalized to [0, 1] against this carrier's track length
- `velocity`       — signed normalized real
- `load_kind`      — empty / pallet-empty / pallet-small / pallet-big (one-hot)
- `busy`           — 0 / 1
- `command_phase`  — moving / giving / taking / handoff-waiting / idle (one-hot)
- `eta`            — normalized real, valid when `busy`
- `is_querying`    — 0 / 1; flags the carrier the policy is currently deciding for

**Shelf nodes.** One per shelf.
- `size_class`     — small / big (one-hot)
- `capacity`       — normalized real
- `depth`          — current stack height, normalized real
- `top_load_kind`  — empty / small / big / none (when shelf is empty) (one-hot)
- `is_transfer`    — 0 / 1
- `is_target`      — 0 / 1; 1 if this shelf holds the item of the
                     highest-priority pending retrieve

**Room nodes.** One per room.
- `ready`          — 0 / 1 (empty pallet present AND serving carrier present)
- `has_pending_store` — 0 / 1
- `pending_store_size` — small / big / none (one-hot)
- `time_since_last_use` — normalized real

**Global node.** A single node with summary features, connected to every
other node so the GNN can broadcast queue context cheaply.
- `n_pending_stores`, `n_pending_retrieves` — normalized counts
- recent arrival-rate estimates from `D̂_t`
- top-of-queue task features (size for stores, target depth for retrieves)

### Edge types

- `accesses` — carrier ↔ shelf, carrier ↔ room (per the static topology)
- `handoff` — carrier ↔ carrier (handoff pose between them)
- `transfer-shelf` — carrier ↔ shelf, carrier ↔ shelf (the shelf appears
                     once; both carriers have edges to it; the shelf's
                     `is_transfer` flag is 1)
- `committed_target` — carrier → target node, present only when the
                       carrier is currently executing a command toward
                       that node. Dynamic.
- `global` — global node ↔ every other node.

All edges are typed; the GNN uses type-conditioned message functions
(R-GCN / heterogeneous GAT). Edge directions matter for `committed_target`
(it points *from* the committing carrier *to* its target).

### Why no absolute IDs

Nothing in the observation references a specific shelf or carrier by ID.
The network sees only types and features and connectivity. The same
trained weights apply to any facility because everything is relative to
the local graph neighborhood and node-level features. This is the property
that makes one trained agent work out of the box on a new facility.

## 3. Action space

At each decision instant, the policy emits exactly one action for the
querying carrier. Actions are typed:

- `take_from(shelf_node)`     — legal iff `shelf_node ∈ accessible_shelves(c)`,
                                shelf non-empty, carrier load = ⊥
- `give_to(shelf_node)`       — legal iff accessible, size-compatible
                                with current load, has capacity, carrier
                                load ≠ ⊥
- `handoff_with(carrier_node)` — legal iff a handoff/narrow-transfer edge
                                 exists, load compatibility holds, the
                                 other carrier is at or heading to the
                                 shared pose
- `move_to_room(room_node)`    — legal iff this carrier serves the room
- `park` / `idle`              — always legal

These are realized as a **single joint softmax** over `(type, target)`
candidate pairs. Concretely the policy:

1. Enumerates every (action_type, candidate_node) pair that is currently
   legal for the querying carrier.
2. Scores each pair with `MLP_type( h_c, h_target )` — one MLP per action
   type, sharing the same input/output shape.
3. Concatenates all scores into one logit vector.
4. Sets illegal pairs' logits to `-∞` (action masking).
5. Softmaxes and samples.

Variable-sized output is native — no fixed action-vector dimension.

## 4. Network architecture

```
input: graph (V, E, node features keyed by type, edge type per edge)

# 4.1 Heterogeneous projection
for each node type t:
    h_v^(0) = MLP_t( features(v) )         # → ℝ^64

# 4.2 K rounds of graph attention (K = 3 default)
for k in 0 .. K-1:
    for each node v:
        # type-conditioned attention scoring per neighbor
        for each neighbor u with edge type e:
            α_vu = softmax_over_neighbors( a_e( W_e [h_v^(k); h_u^(k)] ) )
        h_v^(k+1) = LayerNorm( h_v^(k) + Σ_u α_vu · W_e · h_u^(k) )

# 4.3 Two heads, both consume final embeddings h_v^(K)

# value head
g          = mean_pool({ h_v^(K) : v ∈ V })
V(state)   = MLP_value(g)                 # scalar

# pointer-attention policy head
c          = querying_carrier
logits     = []
for each (type, candidate) in legal_actions(c):
    logits.append( MLP_type(concat(h_c^(K), h_candidate^(K))) )
mask illegal positions to -inf
π(·|state) = Categorical(softmax(logits))
```

Shared trunk: ~50k parameters. Two heads on top, each <10k. Total well
under 100k. PyTorch Geometric or DGL works; rolling your own message-
passing in raw PyTorch is also fine at this size.

## 5. Reward

Per the formalization, per-task cost is `t_completed - t_arrived`. The
training reward is structured to give a dense, well-scaled signal that
sums (in expectation) to the negative of the objective.

**Per-step shaping (continuous).** At every simulator advance of duration
`Δt`, emit:

```
r_step = -Δt × |pending tasks at start of interval|
```

Equivalent in total to summing `-cost(task)` at completion, but credit is
spread across every step a task is alive — the policy gets feedback long
before the task finishes.

**Completion bonus (optional, sparse, small).** A small positive reward
on task completion stabilizes the value head's targets without changing
the optimum: `r_complete = +5`. Pure cosmetic; PPO works fine without it.

**Responsiveness handled by masking, not penalty.** Actions that would
strand a room past its expected next-arrival horizon are masked out at
step 3 of the action head. The policy never sees them as available, so
there's no "soft penalty" gray zone where the agent learns to occasionally
strand rooms. The mask is computed from the current observation +
`D̂_t`'s arrival-rate estimate.

**No reward for reshuffling.** Reshuffling earns reward only via the
downstream effect of making future tasks faster (which shows up as
larger `-Δt × |pending|` reductions later). This is correct: the policy
shouldn't be bribed to reshuffle; it should reshuffle iff doing so
*actually* lowers expected future cost.

## 6. Training loop (PPO)

Standard clipped-surrogate PPO, run against parallel sim workers.

```python
network    = PolicyValueNet(K=3, hidden=64)
optimizer  = Adam(network.parameters(), lr=3e-4)
sim_pool   = ParallelSimWorkers(n_workers=16)

for iteration in range(num_iterations):

    # rollout
    trajectories = sim_pool.collect(
        policy        = network,
        steps         = 2048,                  # per worker
        config_sampler= domain_randomized,
        dist_sampler  = uniform_range_over_rates,
    )

    # advantage estimation (GAE, λ = 0.95, γ = 0.99)
    advantages, returns = gae(trajectories, network.value)

    # multi-epoch minibatch update
    for epoch in range(4):
        for batch in minibatches(trajectories, size=4096):
            dist, value = network(batch.observations)
            log_prob    = dist.log_prob(batch.actions)
            ratio       = torch.exp(log_prob - batch.old_log_prob)

            unclipped   = ratio * batch.advantages
            clipped     = ratio.clamp(1-0.2, 1+0.2) * batch.advantages
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss  = (value - batch.returns).square().mean()
            entropy     = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 0.5)
            optimizer.step()

    log_metrics(...)
    if iteration % checkpoint_interval == 0:
        save(network)
```

Hyperparameters above are conservative defaults known to work for
similar-shape problems. Tune `K` (GNN depth), `hidden`, learning rate,
clip range as needed.

## 7. Domain randomization for generic-out-of-the-box behavior

Each training episode samples:

- **Topology**: number of carriers (2–8), number of shelves per carrier
  (3–20), capacity per shelf (1–6), number of rooms (1–4), connectivity
  pattern (tree, DAG with one cross-link, DAG with multiple cross-links),
  presence and width of transfer shelves.
- **Initial state**: empty-pallet seeding count (sufficient to start
  servicing, but with realistic shortage scenarios occasionally).
- **Task distribution**: per-room store arrival rate, retrieve arrival
  rate, item-size mix on stores, retrieval-likelihood model
  (uniform / recency-skewed / Zipf over historical stores).

The agent never sees the same configuration twice. The GNN trunk forces
the policy to express its behavior in terms of *local graph structure
plus features*, which is exactly the kind of representation that
transfers.

## 8. Deployment and online fine-tuning

At deployment:

1. Snapshot the most recent training checkpoint, freeze it, ship it.
2. The facility's real `D̂_t` is built from observed traffic
   (arrival rates per room, retrieve identity patterns).
3. Every N hours of operation:
   - Log all `(observation, action, reward)` tuples to disk.
   - Run a small fine-tuning pass: a few PPO iterations using the live
     `D̂_t` and a sim that re-plays / resamples from logged traffic.
   - Validate the new checkpoint against the previous one on a held-out
     replay of recent traffic; promote only if it dominates.
4. The fine-tuned checkpoint replaces the running policy live.

Because the GNN trunk is already trained, the fine-tuning is moving only
the last layers' weights against a much narrower distribution. This is
fast (minutes to hours) and stable.

## 9. Baseline to beat

Build a hand-written baseline against the *same* env interface. Minimum
viable baseline:

- Pending tasks served in FIFO order.
- For retrieves: shortest single-carrier path to the target, blockers
  evicted to the nearest legal shelf with free capacity.
- For stores: round-robin over rooms; loaded pallet routed to the nearest
  legal shelf.
- Idle carriers move toward their served room with an empty pallet.

If the RL agent doesn't outperform this on aggregate over thousands of
randomized configs and task streams, it isn't earning its complexity.

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Action mask leaks (illegal action chosen) | Unit-test the mask builder against random states; assert in env that every emitted action is legal pre-application |
| Reward shaping breaks the optimum | Use potential-based shaping only; verify on a small fixed config that PPO with shaping matches PPO with raw sparse reward in optimum |
| Long-horizon credit assignment fails (sparse retrieves) | Per-step `-Δt × |pending|` shaping (above); GAE with λ=0.95; large episode length |
| GNN can't fit / overfits | Start small (K=2, hidden=32); only grow if validation gap is positive |
| Sim-to-real gap | Negligible by construction: sim *is* the deployed code path (`SimFieldProvider`); only the task distribution differs |
| Catastrophic forgetting during fine-tuning | Promote only on held-out replay; KL-regularize fine-tuning toward the shipped checkpoint |
| Configuration distribution at training too narrow | Run a diagnostic suite of "weird" hand-crafted configs (chains of 4 carriers, dense DAGs, transfer-shelf hubs) each iteration; track per-suite reward |

## 11. Minimum viable path

1. **Env wrapper** around `parkolay-sim-engine`: graph builder, action
   applier, reward emitter, event-driven step. ~1 week.
2. **Baseline heuristic** against same interface. ~3 days.
3. **GNN trunk + heads** in PyTorch; supervised proxy task: predict the
   baseline's next action from observation. If the network can imitate
   the baseline, the architecture is sound. ~1 week.
4. **PPO loop** on a fixed config; reward shaping; verify it can beat
   the baseline on that config. ~1–2 weeks.
5. **Domain randomization** turned on; verify generic-config performance.
   ~1–2 weeks.
6. **Deployment harness** for online fine-tuning. ~1 week.

Realistic total to a working v1: 6–9 weeks of focused work.
