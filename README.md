<h2 align="center">reinfocus</h2>

<p align="center">
    <a href="https://github.com/jeffwhunter/reinfocus/blob/main/LICENSE">
        <img
            alt="License: MIT"
            src="https://img.shields.io/badge/license-MIT-blue.svg">
    </a>
    <a href="https://github.com/psf/black">
        <img
            alt="Code style: black"
            src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

`reinfocus` is a python package that makes it easy to create reinforcement learning
environments that use ray tracing to simulate camera focus.

Installation
------------------
To install the base library, use
`pip install git+https://github.com/jeffwhunter/reinfocus`.

For GPU support, install
[the lastest NVIDIA graphics drivers](https://www.nvidia.com/Download/index.aspx). The
next step installs `cudatoolkit` and depends on what type of python installation you use:
 * [anaconda](https://www.anaconda.com/) or
 [variants](https://docs.anaconda.com/free/miniconda/) (recommended): use
 `conda install cudatoolkit`
 * [plain python](https://www.python.org/downloads/release/python-3110/) (untested):
 install [`cudatoolkit`](https://developer.nvidia.com/cuda-downloads), then set
 [`CUDA_HOME`](
    https://numba.readthedocs.io/en/stable/cuda/overview.html#cudatoolkit-lookup)

Only supports Python 3.11 currently.

Examples
------------------
`custom_environments.py` is an example of how you would use `reinfocus` to create various
types of environments that simulate camera focus in different ways.

The rest of the examples need a few extra dependencies; to install them use
`pip install git+https://github.com/jeffwhunter/reinfocus[examples]`.

Some of the examples are [`jupyter`](https://jupyter.org/) files ending in `.ipynb`. They
can be opened by running `jupyter notebook` from the `examples` directory and opening
them. If you're not familiar with jupyter files, just click the 'fast forward' icon at the
top to run the file.

To train an agent with [PPO](https://en.wikipedia.org/wiki/Proximal_policy_optimization)
and watch it's performance in the discrete example environment, use:
 * `python examples\train_agent.py -e DiscreteSteps-v0 -a ppo`
 * `python examples\watch_agent.py -e DiscreteSteps-v0 -a ppo`

To optimize hyperparameters for the same agent and environment (over roughly a day), use:
 * `python examples\optimize_hyperparameters.py -e DiscreteSteps-v0 -a ppo`

To format the generated hyperparameters for `.yml` files, use:
 * `python examples\translate_hyperparameters.py "<paste hyperparameters>"`

Special Thanks
------------------
`reinfocus.graphics` is a [`numba`](
https://numba.readthedocs.io/en/stable/cuda/index.html) translation of the wonderful [Ray
Tracing in One Weekend in CUDA](https://github.com/rogerallen/raytracinginoneweekendincuda
) by [Roger Allan](https://github.com/rogerallen).