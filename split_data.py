'''
This script splits the CIFAR-10 training data's indices such that both validation and training can be performed. 
'''
import numpy as np
from sklearn.model_selection import train_test_split

# Define the number of samples in the CIFAR-10 dataset
num_samples = 50000  # CIFAR-10 has 50,000 training images

# Generate indices
all_indices = np.arange(num_samples)

# Split the indices into training and validation sets
train_indices, val_indices = train_test_split(all_indices, test_size=0.1, random_state=42)

# Save the indices to disk
np.save('data/cifar10_train_indices.npy', train_indices)
np.save('data/cifar10_val_indices.npy', val_indices)

print('Train and validation indices saved!')

