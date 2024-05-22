"""Contains registration details for a number of custom environments."""

from gymnasium.envs import registration


registration.register(
    id="DiscreteSteps-v0",
    entry_point="examples.custom_environments:DiscreteSteps",
    vector_entry_point="examples.custom_environments:VectorDiscreteSteps",
    max_episode_steps=20,
)


registration.register(
    id="ContinuousJumps-v0",
    entry_point="examples.custom_environments:ContinuousJumps",
    max_episode_steps=20,
)
