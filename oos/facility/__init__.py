"""User-facing runtime facility.

For embedding, the public surface is just:

    from oos.facility import Facility

    facility = Facility.from_name("campus")
    obs, info = facility.reset(seed=0)
    while facility.needs_decision():
        action = agent.act(obs, info)
        obs, reward, info = facility.apply_action(action)

`Facility` is the runtime equivalent of the topology-builder
`oos.dsl.Facility` and is implemented on top of the gym env at
`oos.env.env.OOSEnv` — it just removes the gym-shaped term/trunc tuple
and surfaces only what's needed to drive the sim. Episode-level
termination, if you want it, lives in a separate `Episode` wrapper.

Not to be confused with `oos.facilities` (plural) which is the *registry*
of hand-authored topologies (`tiny`, `stacker`, `campus`, ...).
"""

from oos.facility.facility import Facility

__all__ = ["Facility"]
