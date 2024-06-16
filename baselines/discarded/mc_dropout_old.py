import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.classification
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import argparse
import numpy as np
import torchmetrics
import json 
from datetime import datetime
import random

from wide_resnet import WideResNet
from data_utils import load_data


# Add parsing functionality 
parser = argparse.ArgumentParser(description='Wide ResNet with MC Dropout (on CIFAR 10)')

parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='input mini-batch size for training')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=1, type=int, help="seed for reproducibility")
parser.add_argument('--use-scheduler', default=True, type=bool, help="Whether to use a scheduler for the LR or not")
parser.add_argument('--droprate', default=0.3, type=float, help="dropout probability")
parser.add_argument('--subset-size', default=1000, type=int, help="number of data samples used (if None, all used) (for debugging locally)")
parser.add_argument('--forward-passes', default=3, type=int, help="number of MC dropout forward passes for validation")


def set_training_seed(seed):
    # Function to set the different seeds 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, device, train_loader, optimizer, criterion, epoch, scheduler=None):
    print('\nEpoch: %d' % epoch)
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Use LR scheduler 
        if scheduler:
            scheduler.step()

        running_loss += loss.item()
        if batch_idx % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 100)) # print epoch, batch_index and average loss over that mini-batch
            running_loss = 0.0


def validate_mc_dropout(model, test_loader, device, epoch, forward_passes = 5, metrics=None):
    model.train() # Keep the model in training mode to enable dropout during inference
    total = 0
    total_nll = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs_list = []

            # Perform multiple forward passes and store each result
            for i in range(forward_passes):
                outputs =  model(inputs)
                outputs_list.append(F.log_softmax(outputs, dim=1))
                # print(f"Completed forward pass {i} in epoch {epoch}")
            
            # Average the predictions
            outputs_mean = torch.stack(outputs_list).mean(0)
            probs = torch.exp(outputs_mean)  # Convert log probabilities to probabilities
            nll_loss = F.nll_loss(outputs_mean, labels)
            total_nll += nll_loss.item() * inputs.size(0)
            total += labels.size(0)

            if metrics:
                for metric in metrics.values():
                    metric.update(probs, labels)
        
    # Calculate final metric values
    results = {"epoch": epoch, "average_nll": total_nll / total}
    if metrics:
        for name, metric in metrics.items():
            results[name] = metric.compute().item()
            if name == "accuracy":
                results[name] = results[name]*100
            metric.reset()
    
    print(f"The validation results for MC Dropout are as follows: {results}")
    return results

def main():
    # Parse arguments
    args = parser.parse_args(args=[])
    training_seed = args.seed
    batch_size = args.batch_size
    subset_size = args.subset_size 
    data_seed = 42 # seed used for data loading (e.g. transformations)
    num_forward_passes = args.forward_passes

    # Set seed
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # Define metrics to be computed 
    model_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
    model_ECE = torchmetrics.classification.MulticlassCalibrationError(num_classes=10, n_bins=15, norm="l1").to(device)
    metrics = {"accuracy": model_accuracy, "ece": model_ECE}
    
    # Data pre-processing
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size, seed=data_seed, subset_size=subset_size)

    # Set seed for training
    set_training_seed(training_seed)
    
    # Model setup (MC Dropout, so using a droprate (both for training and validation))
    model = WideResNet(depth=28, widen_factor=10, num_classes=10, dropRate=args.droprate).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler (if using one, otherwise None)
    scheduler = None
    if args.use_scheduler:
        print("Now using a scheduler for the LR!!")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    # Results for each epoch
    all_val_results = []
    for epoch in range(args.epochs):
        train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch, scheduler=scheduler)
        val_results = validate_mc_dropout(model=model, test_loader=val_loader, device=device, epoch=epoch, forward_passes=num_forward_passes, metrics=metrics)
        all_val_results.append(val_results)
    
    print(all_val_results)

    # Save results for later
    current_time = datetime.now().strftime("%m-%d-H%H")
    filename = f'mc_results_{current_time}.json'

    with open(filename, 'w') as file:
        json.dump(all_val_results, file)

if __name__ == '__main__':
   main()
    # print("so this works")

