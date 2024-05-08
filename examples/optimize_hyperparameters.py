"""An example of using rl_zoo3 to optimize hyperparameters."""

import argparse
import pathlib
import sys

from typing import Any

import optuna
import yaml

from rl_zoo3 import hyperparams_opt
from rl_zoo3 import train

samplers = {
    "ppo": hyperparams_opt.sample_ppo_params,
    "ppo_lstm": hyperparams_opt.sample_ppo_lstm_params,
}

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
    choices=samplers.keys(),
)

args = parser.parse_args()

examples = pathlib.Path(__file__).parent

sys.path.append(str(examples.parent))

CONFIG_FILENAME = str(examples / f"{args.algo}_untuned.yml")

with open(CONFIG_FILENAME, encoding="utf-8") as config_file:
    continuous = yaml.safe_load(config_file)[args.env].get("use_sde", False)

sys.argv = [
    "python",
    "-optimize",
    "--algo",
    args.algo,
    "--gym-packages",
    "examples.custom_environments",
    "--env",
    args.env,
    "--n-eval-envs",
    "8",
    "--n-evaluations",
    "4",
    "--conf-file",
    CONFIG_FILENAME,
    "--num-threads",
    "20",
    "--n-jobs",
    "20",
    "--device",
    "cuda",
]


def sample_continuous_params(
    trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict
) -> dict[str, Any]:
    """Samples additional hyperparameters for environments with continuous action
    spaces."""

    hyperparams = samplers[args.algo](trial, n_actions, n_envs, additional_args)

    hyperparams.update(
        {
            "sde_sample_freq": trial.suggest_categorical(
                "sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256]
            )
        }
    )
    hyperparams["policy_kwargs"].update(
        {"log_std_init": trial.suggest_float("log_std_init", -4, 1)}
    )

    return hyperparams


if continuous:
    hyperparams_opt.HYPERPARAMS_SAMPLER[args.algo] = sample_continuous_params

train.train()
