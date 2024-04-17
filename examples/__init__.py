"""Contains registration details for a number of custom environments."""

from gymnasium.envs import registration

registration.register(
    id="examples.custom_environments.DiscreteSteps-v0",
    entry_point="examples.custom_environments:DiscreteSteps",
    max_episode_steps=100,
)
