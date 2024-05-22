"""An example of translating a printed hyperparameter string into a format that easily
copies into yml files."""

import argparse
import json

parser = argparse.ArgumentParser(
    prog="python translate_hyperparameters.py",
    description=(
        "Translates the hyperparameters produed by optimize_hyperparameters into a format"
        "suitable for the yml configuration files needed by train_agent"
    ),
)

parser.add_argument(
    "hyperparameters",
    help="A string of hyperparameters copied from optimize_hyperparameters",
    type=str,
)

args = parser.parse_args()

hyperparameters = json.loads(
    str(args.hyperparameters)
    .replace("'", '"')
    .replace("False", "false")
    .replace("True", "true")
)

main_order = [
    "batch_size",
    "n_steps",
    "gamma",
    "learning_rate",
    "ent_coef",
    "sde_sample_freq",
    "clip_range",
    "n_epochs",
    "gae_lambda",
    "max_grad_norm",
    "vf_coef",
]

activation_functions = {"relu": "nn.ReLU", "tanh": "nn.Tanh"}

net_architectures = {
    "tiny": "dict(pi=[64], vf=[64])",
    "small": "dict(pi=[64, 64], vf=[64, 64])",
    "medium": "dict(pi=[256, 256], vf=[256, 256])",
}

for name in main_order:
    if name in hyperparameters:
        print(f"  {name}: {hyperparameters[name]}")

print('  policy_kwargs: "dict(')
print("                    ortho_init=False,")

if "log_std_init" in hyperparameters:
    print(f"                    log_std_init={hyperparameters['log_std_init']},")

activation_function = activation_functions[hyperparameters["activation_fn"]]
print(f"                    activation_fn={activation_function},")

if "lstm_hidden_size" in hyperparameters:
    print(f"                    lstm_hidden_size={hyperparameters['lstm_hidden_size']},")

if "enable_critic_lstm" in hyperparameters:
    enable_critic_lstm = hyperparameters["enable_critic_lstm"]
    print(f"                    enable_critic_lstm={enable_critic_lstm},")

net_architecture = net_architectures[hyperparameters["net_arch"]]
print(f"                    net_arch={net_architecture}")

print('                  )"')
