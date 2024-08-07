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

from data_utils import load_data
from corrupted_data_utils import load_corrupted_data
from wide_resnet import WideResNet

# Add parsing functionality 
parser = argparse.ArgumentParser(description='Evaluate script for model (Deterministic)')

# General arguments
parser.add_argument('--model', type=str, default="placeholder", help='Path of model')

# Function to compute entropy
def compute_entropies(probabilities):
    '''
    Computes the entropy of a set of probabilities. 
    Expected input shape: (batch_size, num_classes)
    '''
    entropies = - (probabilities * torch.log2(probabilities)).sum(dim=1)
    return entropies

def evaluate(model, device, test_loader, dataset="normal", phase="testing"):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    total_nll = 0.0
    num_classes = 10 # CIFAR-10
    ece_metric = CalibrationError(n_bins=15, norm='l1', task="multiclass", num_classes=num_classes).to(device)


    all_entropies = []
    
    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)

            # Calculate NLL loss
            nll_loss = F.cross_entropy(logits, labels, reduction='sum')
            total_nll += nll_loss.item()
            
            probs = torch.softmax(logits, dim=1)

            # Compute accuracy
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()

            # Compute ECE
            ece_metric.update(probs, labels)

            batch_entropies = compute_entropies(probs)
            all_entropies.append(batch_entropies.cpu().numpy())

    # Convert all entropies to a single numpy array
    all_entropies = np.concatenate(all_entropies)

    average_nll = total_nll / total
    accuracy = 100. * correct / total
    ece = ece_metric.compute().item()

    wandb.log({f"{phase}_nll_loss": average_nll, f"{phase}_accuracy": accuracy, f"{phase}_ece": ece})
    
    return all_entropies

def main():
    # Parse arguments
    args = parser.parse_args()

    # Wandb
    run_name = "entropy_exp_DET"
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
    model = WideResNet(depth=28, widen_factor=10, num_classes=10)
    saved_model_det_path = args.model
    model.load_state_dict(torch.load(saved_model_det_path, map_location=torch.device('cpu')))
    model.to(device)

    # Load data
    batch_size = 128
    corrupted_data_loader, normal_data_loader = load_corrupted_data(batch_size=batch_size, seed=5)
    

    # Evaluate the model on the normal data
    corrupted_entropies = evaluate(model, device, corrupted_data_loader, dataset="corrupted", phase="corrupted_testing")
    normal_entropies = evaluate(model, device, normal_data_loader, dataset="normal", phase="testing")

    # Save the lists as separate files within an artifact
    artifact = wandb.Artifact('entropies_lists_det', type='dataset')
    with artifact.new_file('det_normal_data_entropy.txt') as f:
        f.write('\n'.join(map(str, normal_entropies)))
    with artifact.new_file('det_corrupt_data_entropy.txt') as f:
        f.write('\n'.join(map(str, corrupted_entropies)))

    # Log the artifact
    wandb.log_artifact(artifact)

    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    main()