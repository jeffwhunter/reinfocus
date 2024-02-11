"""An example of using rl_zoo3 to optimize hyperparameters."""

import pathlib
import sys

from rl_zoo3 import train

parent = pathlib.Path(__file__).parent

sys.path.append(str(parent))

sys.argv = [
    "python",
    "--algo",
    "ppo_lstm",
    "--gym-packages",
    "custom_environments",
    "--env",
    "ContinuousLeftOrRight-v0",
    "-optimize",
    "--n-trials",
    "100",
    "--sampler",
    "tpe",
    "--pruner",
    "median",
    "-conf",
    str(parent / "ppo_lstm_untuned.yml"),
    "--study-name",
    "Study",
    "--num-threads",
    "10",
    "--n-jobs",
    "10",
]

train.train()
