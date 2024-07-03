'''
This script is a centralized data loading module which creates a consistent corrupted CIFAR-10 dataset for use in 
the various model training scripts. 
'''
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

def set_data_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Define a function to add Gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

# Define the corruption transformations
corruption_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.03, 0.08))], p=0.4),  # Apply Gaussian blur
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.07, hue=0.08, saturation=0.06)], p=0.6),  # Adjust brightness
    transforms.RandomApply([transforms.RandomRotation(degrees=20)], p=0.4),  # Random rotation
    transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.35),  # Random horizontal flip
    transforms.RandomApply([transforms.Resize((22, 22)), transforms.Resize((32, 32))], p=0.3),  # Pixelate
    transforms.RandomApply([transforms.ElasticTransform(alpha=7.0, sigma=2.0)], p=0.5),  # Elastic transformation
    transforms.ToTensor(),  # Ensure the image is a tensor
    transforms.RandomApply([AddGaussianNoise(mean=0.0, std=0.04)], p=0.4),  # Add Gaussian noise
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # Normalize using CIFAR-10 stats
])

normal_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Custom dataset class to apply corruption transformations
class CorruptedCIFAR10(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_corrupted_data(batch_size=128, seed=4000):
    
    set_data_seed(seed)

    # Load CIFAR-10 test data
    cifar10_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())
    cifar10_normal_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=normal_transform)
    
    # Create the corrupted CIFAR-10 test set
    corrupted_cifar10_test = CorruptedCIFAR10(cifar10_test, transform=corruption_transform)

    # Create a DataLoader for the corrupted dataset
    corrupted_test_loader = DataLoader(corrupted_cifar10_test, batch_size=batch_size, shuffle=False)
    normal_test_loader = DataLoader(cifar10_normal_test, batch_size=batch_size, shuffle=False)

    return corrupted_test_loader, normal_test_loader

def visualize_corrupted_data(data_loader):
    batch = next(iter(data_loader))
    images, labels = batch

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        image = images[i+24].permute(1, 2, 0).numpy()
        image = (image * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])).clip(0, 1)  # De-normalize
        axes[i].imshow(image)
        axes[i].axis('off')
    plt.show()

if __name__ == "__main__":
    corrupted_test_loader, normal_test_loader = load_corrupted_data(seed=3)
    visualize_corrupted_data(corrupted_test_loader)
    visualize_corrupted_data(normal_test_loader)