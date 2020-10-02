import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from data import load_fashion_mnist_binary_test
from util.utils import binarise_fashion_mnist, plot_mnist
import gzip



def create_prebinarised_test_data():
    fashion_mnist = input_data.read_data_sets('../../fashion_mnist',
                                              source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
                                              one_hot=True)
    all_test_images, _ = fashion_mnist.test.next_batch(10000)
    binarised_test_images = binarise_fashion_mnist(all_test_images)
    binarised_test_images_int = binarised_test_images.astype(np.int32)
    np.savetxt('data/binarised_fashion_mnist/test_10k.txt', binarised_test_images_int, fmt='%d', newline='\n')
