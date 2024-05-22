Installation
------------------
The examples need some extra dependencies; to install them, use `pip install
reinfocus[examples]`.

To get the examples from github, use:
```
https://github.com/jeffwhunter/reinfocus.git
cd examples
```

Usage
------------------
`custom_environments.py` is an example of how you would use `reinfocus` to create various
types of environments that simulate camera focus in different ways.

To train an agent with [PPO](https://en.wikipedia.org/wiki/Proximal_policy_optimization)
and watch it's performance in the `custom_envionments.DiscreteSteps`, use:
```
python train_agent.py -e DiscreteSteps-v0 -a ppo
python watch_agent.py -e DiscreteSteps-v0 -a ppo
```

The name `DiscreteSteps-v0` is registered with the `custom_environments.DiscreteSteps`
class in `examples\__init__.py`. The hyperparameters used by `train_agent.py` are stored
in `examples\[algo]_tuned.yml`, depending on which algorithm you use. To optimize
hyperparameters yourself (over roughly a day) for the same environment and algorithm, use:
```
python optimize_hyperparameters.py -e DiscreteSteps-v0 -a ppo
```

`optimize_hyperparameters.py` uses some pre-tuned hyperparamers defined in
`examples\[algo]_untuned.yml` to define hyperparameter optimization. Two of these
hyperparameters require some extra attention. First: `policy` must be set to `"MlpPolicy"`
for `ppo` and `"MlpLstmPolicy"` for `ppo_lstm`. Second: `use_sde` must be set to `True`
for any environments with continuous action spaces, like
`custom_environments.ContinuousJumps`. To format the generated hyperparameters for `.yml`
files, use:
```
python translate_hyperparameters.py "<paste hyperparameters>"
```

To use these translated hyperparameters when training an agent for your environment, first
copy that environment's untuned hyperparameters from `examples\[algo]_untuned.yml` to
`examples\[algo]_tuned.yml`, then copy the translated hyperparameters into the same
object. It should look similar to the configurations included in
`examples\[algo]_tuned.yml`. The `n_timesteps` hyperparameter can be increased to increase
the amount of time the agent spends training (ex: change from `!!float 5e4` to
`!!float 5e5` to have the agent train for 10x the length of time each agent was trained
for when optimizing hyperparameters.)

Some of the examples are [`jupyter`](https://jupyter.org/) files ending in `.ipynb`. They
can be opened by running `jupyter notebook` from the `examples` directory and double
clicking them in the UI that opens. If you're not familiar with jupyter files, just click
the 'fast forward' icon at the top to run the file.