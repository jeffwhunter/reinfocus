"""An example of using rl_zoo3 to train an agent."""

import argparse
import pathlib
import sys

from rl_zoo3 import train

algos = {"ppo", "ppo_lstm"}

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", type=str, required=True)
parser.add_argument("-a", "--algo", type=str, required=True, choices=algos)

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
    f"examples.custom_environments.{args.env}",
    "--n-eval-envs",
    "8",
    "--conf-file",
    str(examples / f"{args.algo}_tuned.yml"),
    "--num-threads",
    "20",
    "-P",
    "--tensorboard-log",
    "tensoboard-log/",
]

train.train()
