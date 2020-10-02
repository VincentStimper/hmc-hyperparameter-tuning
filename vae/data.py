import collections
import gzip
import os

import scipy.io as sio
import numpy as np

def load_omniglot(file_path):
    omni_raw = sio.loadmat(file_path)

    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='fortran')

    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
    return train_data, test_data


def load_mnist_raw(file_path, no_shuffle=False):
    if no_shuffle:
        return np.loadtxt(file_path)
    return shuffle_data(np.loadtxt(file_path))


def shuffle_data(data):
    num_data = np.shape(data)[0]
    rand_index = np.random.permutation(np.arange(0, num_data))
    return data[rand_index]


def load_mnist(file_path):
    data = np.load(file_path + '/data.npz')
    Datasets = collections.namedtuple("Datasets", ['train', 'valid', 'test'])
    Datasets.train = MNIST_Data(data['arr_0'])
    Datasets.valid = MNIST_Data(data['arr_1'])
    Datasets.test = MNIST_Data(data['arr_2'], no_shuffle=True)
    return Datasets


class MNIST_Data:
    def __init__(self, data, no_shuffle=False):
        if no_shuffle:
            self.data = data
        else:
            self.data = shuffle_data(data)
        self.num_data_points = np.shape(data)[0]
        self.counter = 0
        self.no_shuffle = no_shuffle

    def next_batch(self, batch_size, init=0):
        if batch_size + self.counter > self.num_data_points:
            if not self.no_shuffle:
                self.data = shuffle_data(self.data)
            self.counter = 0
        mini_batch = self.data[(init+self.counter):(init+self.counter + batch_size)]
        self.counter += batch_size
        return mini_batch, None


def load_iwae_binarised_mnist_dataset(path='./data/binarised_mnist_iwae'):
    if not os.path.exists(path + '/data.npz'):
        print('no npz format found. create npz now.')
        binary_mnist_train = load_mnist_raw(os.path.join(path, 'binarized_mnist_train.amat.txt'))
        binary_mnist_valid = load_mnist_raw(os.path.join(path, 'binarized_mnist_valid.amat.txt'))
        binary_mnist_test = load_mnist_raw(os.path.join(path, 'binarized_mnist_test.amat.txt'))
        np.savez(os.path.join(path, 'data'), binary_mnist_train, binary_mnist_valid, binary_mnist_test)
        print('extraction of the raw data to npz completed.')
    dataset = load_mnist(path)
    return dataset


def create_omniglot_dataset(path):
    train_data, test_data = load_omniglot(path)
    Datasets = collections.namedtuple('Datasets', ['train', 'valid', 'test'])
    Datasets.train = MNIST_Data(train_data)
    Datasets.test = MNIST_Data(test_data)
    return Datasets

def load_mnist_binary_test():
    data = np.load(os.path.join("./data/MNIST_data", 'test_bin.npz'))
    Datasets = collections.namedtuple("Datasets", ['test'])
    Datasets.test = MNIST_Data(data['arr_0'], no_shuffle=True)
    return Datasets

def load_fashion_mnist_binary_test():
    test_data_array = []
    with gzip.open('data/binarised_fashion_mnist/test_10k.txt.gz', 'rt') as f:
        for line in f:
            test_data_array.append(line.split(' '))
    data = np.array(test_data_array).astype(np.float)
    Datasets = collections.namedtuple("Datasets", ['test'])
    Datasets.test = MNIST_Data(data, no_shuffle=True)
    return Datasets
