"""An example of using rl_zoo3 to render a trained agent."""

import pathlib
import sys

from rl_zoo3 import enjoy

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
    "-f",
    "logs/",
    "--exp-id",
    "0",
]

enjoy.enjoy()
