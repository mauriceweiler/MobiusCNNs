import numpy as np
import torch
import torch.utils.data as data
import os.path


class MobiusMnistDataset(data.Dataset):
    """MNIST dataset on the Möbius strip"""
    def __init__(self, mode, shifted=True):
        """
        Args:
            mode (str): Determines whether the train or test set is loaded.
                Must be either of 'train' or 'test'.
            shifted (bool): Whether to load version with shifted or centered digits.

        """
        assert mode in ('train', 'test')
        self.mode = mode

        path = os.path.join('data', 'mobius_MNIST.npz')
        data = np.load(path)

        shift_mode = 'shifted' if shifted else 'centered'
        images = data['{}_data_{}'.format(mode, shift_mode)]
        labels = data['{}_labels'.format(mode)]

        self.images = torch.from_numpy(images).float()
        self.labels = torch.from_numpy(labels).long()
        self.len = len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len


def build_mnist_loader(mode, shifted, batch_size, num_workers=8):
    """Helper function to build an MNIST dataloader

    Args:
        mode (str): Determines whether the train or test set is loaded.
            Must be either of 'train' or 'test'.
        shifted (bool): Whether to load version with shifted or centered digits.
        batch_size (int): Number of elements per batch.
        num_workers (int): Workers of the PyTorch dataloader.

    Returns:
        torch.utils.data.DataLoader: the MNIST dataloader

    """
    if mode == 'train':
        shuffle = True
        drop_last = True
    elif mode == 'test':
        shuffle = False
        drop_last = False
    else:
        raise ValueError('unknown mode')
    dataset = MobiusMnistDataset(mode, shifted=shifted)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    return loader