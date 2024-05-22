"""An example of using rl_zoo3 to render a trained agent."""

import argparse
import pathlib
import sys

from rl_zoo3 import enjoy

algos = {"ppo", "ppo_lstm"}

parser = argparse.ArgumentParser(
    prog="python watch_agent.py",
    description="Load a trained agent and watch it's performance in some environment",
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
parser.add_argument(
    "-i", "--exp-id", help="Which run to watch (defaults to latest)", type=int, default=0
)
parser.add_argument(
    "-b",
    "--best",
    help="Load best agent instead of last",
    action="store_true",
    default=False,
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
    "logs",
    "--exp-id",
    str(args.exp_id),
    "--deterministic",
    "--device",
    "cuda",
]

if args.best:
    sys.argv.append("--load-best")

enjoy.enjoy()
