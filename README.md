# sos-imitation-learning

This repository contains the code from the numerical experiments presented in [Imitation Learning of Stabilizing Policies for Nonlinear Systems](https://www.sciencedirect.com/science/article/pii/S094735802200070X).

The contents of this repository are freely available under the [MIT License](https://choosealicense.com/licenses/mit/).

If this repository or the above paper are useful towards you work, I would greatly appreciate it if you include the citation

```
@article{EAST2022IMITATION,
	title = {Imitation learning of stabilizing policies for nonlinear systems},
	journal = {European Journal of Control},
	pages = {100678},
	year = {2022},
	author = {Sebastian East},
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
|   ├── experts                 # The expert data used for learning (empty at initialization)
|   ├── results                 # The results of the experiments (empty at initializaiton)
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
- CVXPY (v. 1.1.13)
- SCS (v. 2.1.4 - important, see below)
- jax
- matplotlib & seaborn
- Latex (required for generating the plots)

In Ubuntu, a suitable virtual environment can be created with anaconda using
```
conda env create -f environment.yml
```
A suitable environment can similarly be made in other operating systems with modifications to `.yml` (or manually).

The code is *extremely* brittle with respect to the solver used to solve the SOS optimizaiton problems. The code in its present version works well with SCS version 2.1.4, but does not work at all with more recent versions, or the versions of Mosek I have tried. Parsing and solving SOS optimization problems can be numerically challenging, and (to the best of my knowledge) there is no mature open-source SOS framework in Python that can solve this issue. The code has all been tested and works well using the included `environment.yml` file.


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

