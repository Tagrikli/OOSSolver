"""DearPyGui front-end for the facility — a thin, reliable view over the sim.

Three layers, one-way dependency (oos.sim → Session → app):

  * `oos.sim`     — source of truth (state + the motion profile the canvas
                    animates from, via `carrier_position_at`).
  * `Session`     — headless logic: owns the Agent + Environment, drives
                    playback, and exposes the two interaction surfaces
                    (World pokes + Playback). No DearPyGui imports.
  * `app`         — DearPyGui glue: renders a Session, never mutates the sim
                    except through the Session's World/Playback methods.

The rule that keeps it un-buggy: the user mutates the *world* (queue + config);
the *agent* owns the carriers; the canvas only ever renders sim truth at the
current playback time, so the picture cannot drift.
"""

from oos.viz.session import Session

__all__ = ["Session"]
