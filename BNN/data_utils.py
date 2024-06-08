'''
This script is a centralized data loading module which loads the indices for training and validation, which is computed by split_data.py,
and then applies a series of transformations to the dataset. This approach ensures that the same transformations are applied to the same images
across the different model training scripts, promoting reproducibility 
'''
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random

def set_data_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data(batch_size=128, seed=42, subset_size=None):
    '''
    subset_size: number of training samples used (the validation subset size is given by subset_size/2)
    '''

    set_data_seed(seed)

    # Define transformations
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=test_transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)

    # Load indices
    train_indices = np.load('./data/cifar10_train_indices.npy')
    val_indices = np.load('./data/cifar10_val_indices.npy')

    # Optionally create subsets for local debugging
    if subset_size is not None:
        train_indices = np.random.choice(train_indices, subset_size, replace=False)
        val_indices = np.random.choice(val_indices, int(subset_size/2), replace=False)
        batch_size = 32
    
    # Create subsets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)


    # Data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
