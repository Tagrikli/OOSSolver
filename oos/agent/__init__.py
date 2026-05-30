"""Embeddable RL agent — drives an Environment step-by-step.

For embedding in other projects:

    from oos.env import Environment
    from oos.agent import Agent

    env = Environment.from_name("campus")
    agent = Agent.from_checkpoint("runs/.../ckpt_best.pt", facility=env)
    obs, info = agent.reset(seed=0)
    while not agent.done:
        step = agent.step()
        print(step.action_label, step.reward)

No pygame, no toasts, no UI imports.
"""

from oos.agent.agent import Agent, AgentStep, PolicyFn, random_policy

__all__ = ["Agent", "AgentStep", "PolicyFn", "random_policy"]
