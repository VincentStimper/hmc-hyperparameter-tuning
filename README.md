# Tuning hyperparameters of Hamiltonian Monte Carlo

This is the code used to do the experiments of the article "A Gradient Based Strategy for Hamiltonian Monte Carlo Hyperparameter Optimization". 

## Installation instructions

### 2D toy examples

These experiments were done with numpy, scipy, and autograd, so these packages have to be installed.

### Variational Autoencoder

The framework TensorFlow 1.15 was used here. For installation instruction, see https://www.tensorflow.org/install.

### 1D experiments and molecular configuration sampling

We used the framework PyTorch 1.6 for these experiments, see  https://pytorch.org/get-started/locally/ for installation
instructions. Furthermore, to dependencies need to be installed via
```
pip install --upgrade git+https://github.com/VincentStimper/normalizing-flows.git
pip install --upgrade git+https://github.com/VincentStimper/boltzmann-generators.git
```
