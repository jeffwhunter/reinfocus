"""An example of using rl_zoo3 to optimize hyperparameters."""

import argparse
import pathlib
import sys

from typing import Any

import optuna

from rl_zoo3 import hyperparams_opt
from rl_zoo3 import train

samplers = {
    "ppo": hyperparams_opt.sample_ppo_params,
    "ppo_lstm": hyperparams_opt.sample_ppo_lstm_params,
}

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", type=str, required=True)
parser.add_argument("-a", "--algo", type=str, required=True, choices=samplers.keys())
parser.add_argument("-c", "--continuous", action="store_true")

args = parser.parse_args()

examples = pathlib.Path(__file__).parent

sys.path.append(str(examples.parent))

sys.argv = [
    "python",
    "-optimize",
    "--algo",
    args.algo,
    "--gym-packages",
    "examples.custom_environments",
    "--env",
    f"examples.custom_environments.{args.env}",
    "--n-eval-envs",
    "8",
    "--n-evaluations",
    "4",
    "--conf-file",
    str(examples / f"{args.algo}_untuned.yml"),
    "--num-threads",
    "20",
    "--n-jobs",
    "20",
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


if args.continuous:
    hyperparams_opt.HYPERPARAMS_SAMPLER[args.algo] = sample_continuous_params

train.train()
