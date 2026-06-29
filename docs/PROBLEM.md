# The OOS system

An OOS (automated one-piece storage/retrieval) facility is a network of carriers
that move pallets between LIFO shelves and customer-facing rooms. This document
describes **only the system and its dynamics** — the entities, the state, and the
rules by which the state evolves. It deliberately says nothing about what a
controller should try to achieve (no costs, no tasks, no objective); those are
defined separately.

## Entities (static structure)

A facility is fixed at construction time.

### Carriers
- Two kinds: **shuttles** (move along a horizontal track) and **lifts** (move
  along a vertical column).
- Each carrier moves along its own 1-D track between a fixed set of **accessible
  positions**. Motion is trapezoidal — accelerate to a max speed, cruise,
  decelerate — so a move's duration depends on the distance and the carrier's
  motion profile.
- A carrier holds **at most one pallet**.
- Carriers' tracks are pairwise disjoint in space: carriers never collide and
  never block one another. The only coupling between carriers is at handoff
  points (below).

### Shelves
- A shelf is a **LIFO stack** with a fixed **capacity** (number of pallet slots).
  The last pallet pushed on is the first that can be popped off.
- A shelf has a **size class**: *small* (accepts an empty pallet, or a pallet
  carrying a small item) or *big* (accepts an empty pallet, or a pallet carrying
  a small or a big item).
- Most shelves are reachable by exactly one carrier; **transfer shelves** are
  reachable by two.

### Rooms
- A room is a position reachable by exactly one carrier — its **serving carrier**.
  A room is the only place items enter or leave the facility, via customer
  interaction.

### Pallets and items
- **Pallets** are fungible: the system never cares *which* empty pallet sits
  somewhere, only that one does. The total pallet count is conserved for the life
  of the facility — items come and go, pallets do not.
- **Items** have an identity and a **size class** (small or big). An item rides on
  a pallet and is moved only by moving its pallet; an empty pallet carries no item.

### Handoff points (carrier-to-carrier coupling)
- **Handoff pose** — a position on carrier A's track that physically coincides
  with one on carrier B's track. A pallet can pass between A and B only here;
  nothing is stored at a pose.
- **Transfer shelf** — a real shelf reachable by two carriers:
  - *narrow*: behaves like a handoff pose — both carriers must be present to
    exchange a pallet;
  - *wide*: can hold the pallet, so one carrier may leave it and the other take it
    later (the handoff is decoupled in time).

## State (what changes)

At any instant the facility state is:
- for each shelf, its LIFO stack of pallets (each pallet empty or carrying an item);
- for each carrier, its current position and its load (empty, or a pallet that may
  carry an item);
- for each room, what currently sits there (nothing / an empty pallet exposed to
  the customer / a pallet mid-customer-interaction);
- a continuous clock.

Initially the carriers sit at default positions empty, the shelves hold a
pre-seeded distribution of (mostly empty) pallets, and no customer interaction is
in progress.

## Dynamics (how the state evolves)

The state changes through carrier actions and through exogenous customer
interactions.

### Carrier primitives
The control interface chooses, per carrier, one primitive at a time. A carrier is
asked for a decision only while it is idle (not mid-motion). The primitives:

- **GOTO(target)** — move the carrier to one of its accessible locations: a shelf
  position, a room, or a handoff pose. Duration follows the distance and the
  carrier's motion profile.
- **TAKE** — pop the top pallet of the shelf the carrier is at onto the carrier.
  Requires the carrier empty and the shelf non-empty.
- **GIVE** — push the carrier's held pallet onto the top of the shelf it is at.
  Requires size compatibility and free capacity.
- **WAIT** — stay idle until some state change re-opens a decision.

A TAKE or GIVE is a short, fixed-duration pick/place.

### Handoffs (automatic on rendezvous)
When two partner carriers are simultaneously at a matching handoff pose (or narrow
transfer shelf), one loaded and one empty, the pallet transfers from the loaded to
the empty carrier **automatically** — there is no separate handoff command; a
carrier simply GOTOs the pose and the transfer fires the instant its partner is
there. A *wide* transfer shelf instead uses ordinary GIVE/TAKE, decoupled in time.

### Customer interactions at rooms (how items enter and leave)
A customer interacts with the pallet sitting at a room:
- **load** — the customer places an item onto an empty pallet that the serving
  carrier has staged at the room; the pallet now carries that item and the carrier
  can move it into the shelves;
- **unload** — the customer removes an item from a pallet that has been brought to
  the room; the item leaves the facility and the pallet becomes empty.

A room is **ready** for a load when its serving carrier is parked at it holding an
empty pallet.

### Concurrency
Carriers act in parallel and independently. The only synchronization is the
rendezvous at a handoff pose or narrow transfer shelf, which requires both involved
carriers to be present at the same instant; a wide transfer shelf is a shared
buffer that decouples them in time. There is no global lock — one carrier working
never freezes the others.

### What makes the dynamics non-trivial
- Shelves are LIFO, so reaching a pallet buried under others first requires moving
  the pallets above it elsewhere.
- Routing a pallet between regions that no single carrier spans requires one or
  more handoffs, each a synchronization point between two carriers.
- Size classes constrain where a pallet may be placed.
