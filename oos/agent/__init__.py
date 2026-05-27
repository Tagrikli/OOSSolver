"""Embeddable RL agent — drives a Facility step-by-step.

For embedding in other projects:

    from oos.facility import Facility
    from oos.agent import Agent

    facility = Facility.from_name("campus")
    agent = Agent.from_checkpoint("runs/.../ckpt_best.pt", facility=facility)
    obs, info = agent.reset(seed=0)
    while not agent.done:
        step = agent.step()
        print(step.action_label, step.reward)

No pygame, no toasts, no UI imports.
"""

from oos.agent.agent import Agent, AgentStep, PolicyFn, random_policy

__all__ = ["Agent", "AgentStep", "PolicyFn", "random_policy"]
