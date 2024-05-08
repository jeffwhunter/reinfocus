"""An example of using rl_zoo3 to render a trained agent."""

import argparse
import pathlib
import sys

from rl_zoo3 import enjoy

algos = {"ppo", "ppo_lstm"}

parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--env",
    help="The environment in which the agents will act",
    type=str,
    required=True,
)
parser.add_argument(
    "-a",
    "--algo",
    help="The algorithms that will train the agents",
    type=str,
    required=True,
    choices=algos,
)
parser.add_argument(
    "-i", "--exp-id", help="Which run to watch (defaults to latest)", type=int, default=0
)

args = parser.parse_args()

examples = pathlib.Path(__file__).parent

sys.path.append(str(examples.parent))

sys.argv = [
    "python",
    "--algo",
    args.algo,
    "--gym-packages",
    "examples.custom_environments",
    "--env",
    args.env,
    "-f",
    "logs/",
    "--exp-id",
    args.exp_id,
    "--deterministic",
    "--load-best",
    "--device",
    "cuda",
]

enjoy.enjoy()
