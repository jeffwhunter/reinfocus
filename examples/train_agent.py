"""An example of using rl_zoo3 to train an agent."""

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
    "-conf",
    str(parent / "ppo_lstm_tuned.yml"),
    "-P",
    "--num-threads",
    "10",
    "--tensorboard-log",
    "tensoboard-log/",
]

train.train()
