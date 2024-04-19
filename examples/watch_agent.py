"""An example of using rl_zoo3 to render a trained agent."""

import argparse
import pathlib
import sys

from rl_zoo3 import enjoy

algos = {"ppo", "ppo_lstm"}

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", type=str, required=True)
parser.add_argument("-a", "--algo", type=str, required=True, choices=algos)
parser.add_argument("-i", "--exp-id", type=int, default=0)

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
]

enjoy.enjoy()
