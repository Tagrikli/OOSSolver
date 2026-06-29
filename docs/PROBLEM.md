# The OOS system

An OOS facility is an automated car storage/retrieval system — an automated valet
car park. It is a network of carriers that move **pallets** (each holding at most
one car) between LIFO shelves and customer-facing rooms. This document describes
**only the system and its dynamics** — the entities, the state, and the rules by
which the state evolves. It deliberately says nothing about what a controller
should try to achieve (no costs, no tasks, no objective); those are defined
separately.

## Entities (static structure)

A facility is fixed at construction time.

### Carriers

- Two kinds: **shuttles** (move along a horizontal track) and **lifts** (move
  along a vertical column).
- Each carrier moves along its own 1-D track between a fixed set of **accessible
  positions**. Motion is trapezoidal — accelerate to a max speed, cruise,
  decelerate — so a move's duration depends on the distance and the carrier's
  motion profile.
- A carrier holds **at most one pallet** (and therefore at most one car).
- Carriers' tracks are pairwise disjoint in space: carriers never collide and
  never block one another. The only coupling between carriers is at handoff
  points (below).

### Shelves

- A shelf is a **LIFO stack** with a fixed **capacity** (number of pallet slots).
  The last pallet pushed on is the first that can be popped off.
- A shelf has a **size class**: *sedan* or *SUV*. A **sedan** shelf accepts only an
  empty pallet or a pallet carrying a sedan. An **SUV** shelf accepts anything — an
  empty pallet, a sedan, or an SUV. (A sedan fits on any shelf; an SUV fits only on
  SUV shelves.)
- Most shelves are reachable by exactly one carrier; **transfer shelves** are
  reachable by two.

### Rooms

- A room is a position reachable by exactly one carrier — its **serving carrier**.
  A room is the only place cars enter or leave the facility, through customer
  interaction (see Dynamics).

### Pallets and cars

- **Pallets** are fungible: the system never cares *which* empty pallet sits
  somewhere, only that one does. The total pallet count is conserved for the life
  of the facility — cars come and go, pallets do not.
- **Cars** are what the facility stores. Each car has an identity (a customer will
  later ask for that specific car) and a **size class** — *sedan* or *SUV*. A car
  always sits **on a pallet** and is never handled directly: it is moved only by
  moving the pallet beneath it. An empty pallet carries no car.

### Handoff points (carrier-to-carrier coupling)

- **Handoff pose** — a position on carrier A's track that physically coincides
  with one on carrier B's track. A pallet can pass between A and B only here;
  nothing is stored at a pose.
- **Transfer shelf** — a real shelf reachable by two carriers:
  - *narrow*: behaves like a handoff pose — both carriers must be present to
    exchange a pallet;
  - *wide*: can hold the pallet, so one carrier may leave it and the other take it
    later (the handoff is decoupled in time).

## Facility layouts

The entities above are common to every facility, but how many of each there are —
and how they are wired together — varies widely, from a single carrier with one
room up to many carriers, many rooms, and several handoff layers. What varies:

- **Rooms — one or many.** A facility has at least one room (cars enter and leave
  only through rooms) but may have several.
- **Each room has exactly one serving carrier; a serving carrier may serve one or
  several rooms.** The room → carrier mapping is many-to-one.
- **Serving vs. non-serving carriers.** A carrier that serves at least one room is
  a *serving carrier* — it stages and delivers at its room(s) and also tends its
  own shelves. A carrier that serves no room is a *non-serving carrier*: it owns
  its own shelves and only relocates pallets and takes part in handoffs; it never
  reaches a room. A facility may have any number of non-serving carriers,
  including none.

These dimensions span, for example:

- **single carrier, single room** — the minimal facility: one carrier that stages
  and serves its single room and owns all the shelves it can reach.
- **single carrier, multiple rooms** — one carrier responsible for several rooms,
  plus its shelves.
- **multiple rooms, multiple serving carriers, and multiple non-serving carriers**
  — the general, and most important, case: several rooms, each with its own serving
  carrier, together with non-serving carriers that shuttle pallets between shelf
  regions and hand them off to the serving carriers. Here a single store or
  retrieve routinely crosses carrier boundaries, so handoffs and transfer shelves
  do the heavy lifting.

## State (what changes)

At any instant the facility state is:

- for each shelf, its LIFO stack of pallets (each pallet empty or carrying a car);
- for each carrier, its current position and its load (empty, or a pallet that may
  carry a car);
- for each room, what currently sits there (nothing / an empty pallet staged for a
  customer / a pallet mid-customer-interaction);
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
  Requires size compatibility (an SUV only onto an SUV shelf) and free capacity.
- **WAIT** — stay idle until some state change re-opens a decision.

A TAKE or GIVE is a short, fixed-duration pick/place.

### Handoffs (automatic on rendezvous)

When two partner carriers are simultaneously at a matching handoff pose (or narrow
transfer shelf), one loaded and one empty, the pallet transfers from the loaded to
the empty carrier **automatically** — there is no separate handoff command; a
carrier simply GOTOs the pose and the transfer fires the instant its partner is
there. A *wide* transfer shelf instead uses ordinary GIVE/TAKE, decoupled in time.

### Customer interactions at rooms (how cars enter and leave)

A room is **staged** when its serving carrier is parked at it holding an **empty
pallet**. A staged room is open for a customer:

- **store (park a car)** — a customer drives any car, sedan or SUV, onto the empty
  pallet staged at the room. The pallet now carries that car, and the serving
  carrier takes it away to store it on a size-compatible shelf somewhere in the
  facility.
- **retrieve (pick up a car)** — to return a previously stored car, its serving
  carrier brings the pallet carrying that *specific* car to the room; the customer
  drives the car off, the car leaves the facility, and the pallet becomes empty.

Both interactions are customer-driven (exogenous). The system's role is to keep
rooms staged so a customer can park, and to deliver the right pallet-and-car to a
room so a customer can pick up.

### Concurrency

Carriers act in parallel and independently. The only synchronization is the
rendezvous at a handoff pose or narrow transfer shelf, which requires both involved
carriers to be present at the same instant; a wide transfer shelf is a shared
buffer that decouples them in time. There is no global lock — one carrier working
never freezes the others.

### What makes the dynamics non-trivial

- Shelves are LIFO, so reaching a car buried under other pallets first requires
  moving the pallets above it elsewhere.
- Routing a pallet between regions that no single carrier spans requires one or
  more handoffs, each a synchronization point between two carriers.
- Size classes constrain placement: an SUV can only rest on an SUV shelf; a sedan
  fits anywhere.
