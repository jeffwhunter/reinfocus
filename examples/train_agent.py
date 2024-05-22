"""An example of using rl_zoo3 to train an agent."""

import argparse
import pathlib
import sys

from rl_zoo3 import train

algos = {"ppo", "ppo_lstm"}

parser = argparse.ArgumentParser(
    prog="python train_agent.py",
    description=(
        "Trains an agent to perform in some environment and saves snapshots of the agent"
        "over the training process"
    ),
)

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
    "--n-eval-envs",
    "8",
    "--conf-file",
    str(examples / f"{args.algo}_tuned.yml"),
    "--num-threads",
    "20",
    "-P",
    "--tensorboard-log",
    "tensorboard-log/",
    "--device",
    "cuda",
]

train.train()
