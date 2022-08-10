# Tuning hyperparameters of Hamiltonian Monte Carlo

This is the code used to do the experiments of the article [A Gradient Based Strategy for Hamiltonian Monte Carlo
Hyperparameter Optimization](https://proceedings.mlr.press/v139/campbell21a.html). 

## Installation instructions

### 2D toy examples

These experiments were done with numpy, scipy, and autograd, so these packages have to be installed.

### Variational Autoencoder

The framework TensorFlow 1.15 was used here. For installation instruction, see https://www.tensorflow.org/install.

### 1D experiments and molecular configuration sampling

We used the framework PyTorch 1.6 for these experiments, see  https://pytorch.org/get-started/locally/ for installation
instructions. The experiments involving alanine dipeptide require [OpenMM](https://openmm.org/) to be
installed, which can be done via [conda](https://anaconda.org/conda-forge/openmm). The other dependencies
can be installed via

```
pip install -r requirements.txt
```

The scripts for running the experiments are in the `molecular-configurations` directory. Each experiment can be
reproduced using the respective configuration file in `molecular-configurations/config`.