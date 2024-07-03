import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import argparse
import numpy as np
import torchmetrics
from torchmetrics.classification import CalibrationError

import random
import wandb


from rank1_wide_resnet import Rank1Bayesian_WideResNet
from data_utils import load_data
from corrupted_data_utils import load_corrupted_data

# Add parsing functionality 
parser = argparse.ArgumentParser(description='Evaluate script for model (BNN)')

# General arguments
parser.add_argument('--model', type=str, default="placeholder", help='Path of model')
parser.add_argument('--num-eval-samples', default=10, type=int, help="Number of samples to use for evaluation")
parser.add_argument('--ensemble-size', default=4, type=int, help="Number of models in the ensemble")


# Function to compute entropy
def compute_entropies(probabilities):
    '''
    Computes the entropy of a set of probabilities. 
    Expected input shape: (batch_size, num_classes)
    '''
    entropies = - (probabilities * torch.log(probabilities)).sum(dim=1)
    return entropies

def evaluate(model, device, test_loader, num_eval_samples, dataset="normal"):
    model.eval()
    num_classes = 10 # CIFAR-10

    all_entropies = []
    
    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Handle multiple samples
            logits = torch.stack([model(inputs) for _ in range(num_eval_samples)], dim=2) # Shape: (batch_size*ensemble_size, num_classes, num_eval_samples)
            logits = logits.view(model.ensemble_size, -1, num_classes, num_eval_samples) # Shape: (ensemble_size, batch_size, num_classes, num_eval_samples)
            logits = logits.permute(1, 2, 0, 3) # Shape: (batch_size, num_classes, ensemble_size, num_eval_samples)
            probs = torch.softmax(logits, dim=1)

            # Average probs over ensemble_size and num_eval_samples, make predictions and compute accuracy
            mean_probs = probs.mean(dim=(2, 3)) # Shape: (batch_size, num_classes)

            batch_entropies = compute_entropies(mean_probs)
            all_entropies.append(batch_entropies.cpu().numpy())

    # Convert all entropies to a single numpy array
    all_entropies = np.concatenate(all_entropies)
    return all_entropies

def main():
    # Parse arguments
    args = parser.parse_args()

    # Wandb
    run_name = "entropy_exp_BNN"
    wandb.init(project='evaluation_only', mode="online", name=run_name)
    wandb.config.update(args)

     # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # Load model
    model = Rank1Bayesian_WideResNet(depth=28, widen_factor=10, num_classes=10, ensemble_size=args.ensemble_size)
    saved_model_bnn_path = args.model
    model.load_state_dict(torch.load(saved_model_bnn_path, map_location=torch.device('cpu')))
    model.to(device)

    # Load data
    batch_size = 128
    corrupted_data_loader, normal_data_loader = load_corrupted_data(batch_size=batch_size, seed=5)
    
    # Define the number of evaluation samples to test
    num_eval_samples = args.num_eval_samples

    # Evaluate the model on the normal data
    corrupted_entropies = evaluate(model, device, corrupted_data_loader, num_eval_samples, dataset="corrupted")
    normal_entropies = evaluate(model, device, normal_data_loader, num_eval_samples, dataset="normal")

    # Save the lists as separate files within an artifact
    artifact = wandb.Artifact('entropies_lists', type='dataset')
    with artifact.new_file('BNN_normal_data_entropy.txt') as f:
        f.write('\n'.join(map(str, normal_entropies)))
    with artifact.new_file('BNN_corrupt_data_entropy.txt') as f:
        f.write('\n'.join(map(str, corrupted_entropies)))

    # Log the artifact
    wandb.log_artifact(artifact)

    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    main()