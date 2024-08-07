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


from wide_resnet import WideResNet
from data_utils import load_data
from corrupted_data_utils import load_corrupted_data

parser = argparse.ArgumentParser(description='Evaluate script for Deep Ensembles')

parser.add_argument('--model1', type=str, default="placeholder", help='Path of model')
parser.add_argument('--model2', type=str, default="placeholder", help='Path of model')
parser.add_argument('--model3', type=str, default="placeholder", help='Path of model')
parser.add_argument('--model4', type=str, default="placeholder", help='Path of model')
parser.add_argument('--run-name', type=str, default="ens_1234", help='Name of run')


# Function to compute entropy
def compute_entropies(probabilities):
    '''
    Computes the entropy of a set of probabilities. 
    Expected input shape: (batch_size, num_classes)
    '''
    entropies = - (probabilities * torch.log2(probabilities)).sum(dim=1)
    return entropies


# Function to load models
def load_models(model_paths, device):
    models = []
    for path in model_paths:
        model = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()  # Set to evaluation mode
        models.append(model)
    return models

def evaluate(models, device, test_loader, dataset="normal"):
    correct = 0
    total = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    total_nll = 0
    num_classes = 10 # CIFAR-10
    ece_metric = CalibrationError(n_bins=15, norm='l1', task="multiclass", num_classes=num_classes).to(device)
    all_entropies = []

   
    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)

            logits = torch.stack([models[0](inputs), models[1](inputs), models[2](inputs), models[3](inputs)], dim=2)
            probs = torch.softmax(logits, dim=1)

            
            # Duplicate labels for the ensemble size and num_eval_samples
            labels_expanded = labels.unsqueeze(1).expand(-1, len(models)) # Shape: (batch_size, num_forward_passes)
         
            # Compute the log likelihoods
            log_likelihoods = - F.cross_entropy(logits, labels_expanded, reduction="none") # Shape: (batch_size, num_forward_passes)
            logsumexp_temp = - torch.logsumexp(log_likelihoods, dim=1) + math.log(len(models)) # Eq. 14 in the paper

            # Return the mean NLL across the batch 
            nll = logsumexp_temp.mean() 
            total_nll += nll.item()

            # Average probs over ensemble_size and num_eval_samples, make predictions and compute accuracy
            mean_probs = probs.mean(dim=2) # Shape: (batch_size, num_classes)
            preds = mean_probs.argmax(dim=1) 
            correct += preds.eq(labels).sum().item()

            # Compute ECE
            ece_metric.update(mean_probs, labels)

            # Compute entropies
            batch_entropies = compute_entropies(mean_probs)
            all_entropies.append(batch_entropies.cpu().numpy())


    average_nll = total_nll / len(test_loader)
    accuracy = 100. * correct / total
    ece = ece_metric.compute().item()
    
    wandb.log({f"{dataset}_average_nll": average_nll, f"{dataset}_accuracy": accuracy, f"{dataset}_ece": ece})

    all_entropies = np.concatenate(all_entropies)

    return accuracy, average_nll, ece, all_entropies

def main():
    # Parse arguments
    args = parser.parse_args()

    # Wandb
    run_name = args.run_name
    wandb.init(project='ensemble', mode="online", name=run_name)
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
    model_paths = [args.model1, args.model2, args.model3, args.model4]
    models = load_models(model_paths, device)

    # Load data
    batch_size = 128
    corrupted_data_loader, normal_data_loader = load_corrupted_data(batch_size=batch_size, seed=5)
    
  

    # Evaluate the model on the normal data
    
    accuracy, average_nll, ece, normal_entropies = evaluate(models, device, normal_data_loader, dataset="normal")
    print(f"Normal data: Accuracy: {accuracy}, Average NLL: {average_nll}, ECE: {ece}")
    
    # Evaluate the model on the corrupted data
    
    accuracy, average_nll, ece, corrupted_entropies = evaluate(models, device, corrupted_data_loader, dataset="corrupted")
    print(f"Corrupted data: Accuracy: {accuracy}, Average NLL: {average_nll}, ECE: {ece}")


    # Save the lists as separate files within an artifact
    artifact = wandb.Artifact('entropies_lists_ens', type='dataset')
    with artifact.new_file('ens_normal_data_entropy.txt') as f:
        f.write('\n'.join(map(str, normal_entropies)))
    with artifact.new_file('ens_corrupt_data_entropy.txt') as f:
        f.write('\n'.join(map(str, corrupted_entropies)))

    # Log the artifact
    wandb.log_artifact(artifact)

    # Finish the run
    wandb.finish()

if __name__ == "__main__":
    main()
