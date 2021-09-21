# sos-imitation-learning

This repository contains the code from the numerical experiments presented in [Imitation Learning of Stabilizing Policies for Nonlinear Systems](https://arxiv.org/pdf/2001.02244.pdf) (currently under review).

The contents of this repository are freely available under the [MIT License](https://arxiv.org/pdf/2001.02244.pdf).

If this repository, or the above paper, are useful towards you work, I would greatly appreciate it if you include the citation

```
@ARTICLE{East2021,
       author = {{East}, Sebastian,
        title = "{Imitation Learning of Stabilizing Policies for Nonlinear Systems}",
      journal = {arXiv e-prints},
 primaryClass = {math.OC},
}
```

in your relevant publications.

## Repository Structure

```
.
├── config                  # Configuration files for experiments
|   ├── schedule.py             # The testing schedule
|   └── systems.py              # The systems and controllers to be tested
├── data                    # Folders for the data generated during experiements (empty at initialization)
|   ├── experts                 # The expert data used for learning
|   ├── results                 # The results of the experiments
|   └── figures                 # The figures generated for publication
├── docs                    # Documentation of contents of ./sos (empty at initialization)
├── sos                     # The sum of squares analysis tools used in the experiments
|   ├── data_generation.py      # Generating data for experiments
|   ├── sos.py                  # Sum of squares tools for system analysis and imitation learning
|   └── validation              # Tools for validating the output of the sos tools
├── test                    # Unit & integration tests
└── ...                     # Files for simulation and for creating docs
```

## Requirements

The code in this repository requires

- numpy
- sympy
- scipy
- CVXPY
- SCS
- jax
- matplotlib & seaborn

In Linux, a suitable virtual environment can be created with anaconda using
```
conda env create .yml
```
A suitable environment can similarly be made in other operating systems with modifications to `.yml` (or manually).

## Doc creation

In Linux, documentation for the contents of 'sos.py' can be created with `pydoc` using
```
bash makedocs.sh
```
Documentation in other operating systems can be created using similar commands.


## Sum of squares systems analysis

The package in `./sos` can be used for both imitation learning and SOS analysis of nonlinear systems. The script `analysis.py` can be used to generate Lyapunov functions for both of the tests systems when controlled by their 'experts', and to determine whether the imitation learning process is feasable.

## Running the Experiments

In Linux, the experiments can be run end-to-end with the command
```
bash run.sh
```
Essentially, this sequentially runs the following four python scripts:
```
generate_data.py              # Generates the 'expert' data, and saves it in ./data/experts
run_experiments.py            # Runs the experiments, and saves the results in ./data/results
plot_nonlinear_system.py      # Generates the publication plots for the nonlinear system, and saves them in ./data/figures
plot_nonlinear_controller.py  # Same for the nonlinear controller
```

