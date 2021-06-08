import numpy as np
from torchvision import datasets
import os.path

from utils import isom_action_numpy


def gen_mobius_mnist(train_size=12000):
    """Generates MNIST datasets on the Mobius strip and saves it as .npz file

    Args:
        train_size (int): Number of digits in the training set.
            By default set to 12k in compliance with the rotated MNIST dataset.
            Maximally 60k, which is the size of the original MNIST training set.

    """
    trainset = datasets.MNIST(root='data', train=True,  download=True)
    testset  = datasets.MNIST(root='data', train=False, download=True)
    assert train_size <= len(trainset)
    train_data_shifted = []
    train_data_centered = []
    train_labels = []
    test_data_shifted = []
    test_data_centered = []
    test_labels = []
    for i,(image,label) in enumerate(trainset):
        if i == train_size:
            break
        image = np.array(image)[np.newaxis] / 255.
        shift = np.random.randint(2*image.shape[-1])
        shifted = isom_action_numpy(image, shift, (1,0,0))
        train_data_shifted.append(shifted)
        train_data_centered.append(image)
        train_labels.append(label)
    for image,label in testset:
        image = np.array(image)[np.newaxis] / 255.
        shift = np.random.randint(2*image.shape[-1])
        # shifted = mobius_isometry_action_triv(image, shift)
        shifted = isom_action_numpy(image, shift, (1,0,0))
        test_data_shifted.append(shifted)
        test_data_centered.append(image)
        test_labels.append(label)
    path = os.path.join('data', 'mobius_MNIST.npz')
    np.savez(path, train_data_shifted=train_data_shifted,
                   train_data_centered=train_data_centered,
                   train_labels=train_labels,
                   test_data_shifted=test_data_shifted,
                   test_data_centered=test_data_centered,
                   test_labels=test_labels)


if __name__ == '__main__':
    gen_mobius_mnist() # download and generate dataset