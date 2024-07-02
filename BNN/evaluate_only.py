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
parser.add_argument('--ensemble-size', default=2, type=int, help="Number of models in the ensemble")


def evaluate(model, device, test_loader, num_eval_samples, dataset="normal"):
    model.eval()
    correct = 0
    total = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    total_nll = 0
    num_classes = 10 # CIFAR-10
    ece_metric = CalibrationError(n_bins=15, norm='l1', task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Handle multiple samples
            logits = torch.stack([model(inputs) for _ in range(num_eval_samples)], dim=2) # Shape: (batch_size*ensemble_size, num_classes, num_eval_samples)
            logits = logits.view(model.ensemble_size, -1, num_classes, num_eval_samples) # Shape: (ensemble_size, batch_size, num_classes, num_eval_samples)
            logits = logits.permute(1, 2, 0, 3) # Shape: (batch_size, num_classes, ensemble_size, num_eval_samples)
            probs = torch.softmax(logits, dim=1)

            # Duplicate labels for the ensemble size and num_eval_samples
            labels_expanded = labels.unsqueeze(1).unsqueeze(2).expand(-1, model.ensemble_size, num_eval_samples) # Shape: (batch_size, ensemble_size, num_eval_samples)

            # Compute the log likelihoods
            log_likelihoods = - F.cross_entropy(logits, labels_expanded, reduction="none") # Shape: (batch_size, ensemble_size, num_eval_samples)
            logsumexp_temp = - torch.logsumexp(log_likelihoods, dim=(1, 2)) + math.log(model.ensemble_size * num_eval_samples) # Eq. 14 in the paper
            
            # Return the mean NLL across the batch 
            nll = logsumexp_temp.mean() 
            total_nll += nll.item()

            # Average probs over ensemble_size and num_eval_samples, make predictions and compute accuracy
            mean_probs = probs.mean(dim=(2, 3)) # Shape: (batch_size, num_classes)
            preds = mean_probs.argmax(dim=1) 
            correct += preds.eq(labels).sum().item()

            # Compute ECE
            ece_metric.update(mean_probs, labels)


    average_nll = total_nll / len(test_loader)
    accuracy = 100. * correct / total
    ece = ece_metric.compute().item()
    
    wandb.log({f"{dataset}_average_nll": average_nll, f"{dataset}_accuracy": accuracy, f"{dataset}_ece": ece})

    return accuracy, average_nll, ece

def main():
    # Parse arguments
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using CUDA
    np.random.seed(seed)
    random.seed(seed)

    # Wandb
    run_name = args.model
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
    evaluation_samples = [1, 1, 1, 2, 2, 2]

    # Evaluate the model on the normal data
    for num_eval_samples in evaluation_samples:
        accuracy, average_nll, ece = evaluate(model, device, normal_data_loader, num_eval_samples, dataset="normal")
        print(f"Normal data: Accuracy: {accuracy}, Average NLL: {average_nll}, ECE: {ece}")
    
    # Evaluate the model on the corrupted data
    for num_eval_samples in evaluation_samples:
        accuracy, average_nll, ece = evaluate(model, device, corrupted_data_loader, num_eval_samples, dataset="corrupted")
        print(f"Corrupted data: Accuracy: {accuracy}, Average NLL: {average_nll}, ECE: {ece}")

if __name__ == "__main__":
    main()
