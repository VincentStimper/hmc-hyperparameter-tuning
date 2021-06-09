# Compressed Sensing Experiments

## Dependencies
 - https://github.com/VincentStimper/normalizing-flows
 - https://github.com/mfouesneau/NUTS
 - https://github.com/facebookresearch/hydra

## Usage
First generate data
```
python generate_train_and_test_data.py
```
Train initial distribution
```
python train_initial_distribution.py
```
Run maxELT
```
python train_maxELT.py
```
To evaluate run
```
python evaluate_model.py
```
with appropriate settings in config file

For baselines
```
python grid_search.py
python train_acc_prob.py
python run_nuts.py
```