import numpy as np
import torch
import hydra
import subprocess
from core.data_generation import generate_data


@hydra.main(config_path='conf', config_name="data_generation")
def main(cfg):

    num_points_to_generate = cfg.num_training_datapoints + \
        cfg.num_validation_datapoints + cfg.num_test_datapoints

    X, y, w, lamb = generate_data(num_points_to_generate, cfg.w_dim,
        cfg.tau, cfg.sigma_0)

    n1 = cfg.num_training_datapoints
    n2 = cfg.num_validation_datapoints
    n3 = cfg.num_test_datapoints

    X_train = X[0:n1, :]
    y_train = y[0:n1]
    X_validation = X[n1:n1+n2,:]
    y_validation = y[n1:n1+n2]
    X_test = X[n1+n2:n1+n2+n3,:]
    y_test = y[n1+n2:n1+n2+n3]

    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_validation.npy', X_validation)
    np.save('y_validation.npy', y_validation)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    np.save('w.npy', w)
    np.save('lambda.npy', lamb)

if __name__ == "__main__":
    main()
