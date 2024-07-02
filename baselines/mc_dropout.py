import torch
import math
import torch.nn as nn
import torch.optim as optim
import torchmetrics.classification
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import argparse
import numpy as np
import torchmetrics
from torchmetrics.classification import CalibrationError
import json 
from datetime import datetime
import random
import wandb 

from wide_resnet_mc_dropout import WideResNet

from data_utils import load_data
from custom_scheduler import WarmUpPiecewiseConstantSchedule

# Add parsing functionality 
parser = argparse.ArgumentParser(description='MC dropout Wide ResNet (on CIFAR 10)')

parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='input mini-batch size for training')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=1, type=int, help="seed for reproducibility")
parser.add_argument('--use-scheduler', default=True, type=bool, help="Whether to use a scheduler for the LR or not")
parser.add_argument('--use-subset', default=False, type=bool, help="whether to use a subset (for debugging locally) or all data")
parser.add_argument('--wandb', default="online", type=str, choices=["online", "disabled"] , help="whether to track with weights and biases or not")
parser.add_argument('--warmup-epochs', default=1, type=int, help="Number of warmup epochs")
parser.add_argument('--scheduler', default="warm", type=str, choices=["warm", "cosine", "multistep", "none"], help="which scheduler to use")

# MC Dropout specific arguments
parser.add_argument('--droprate', default=0.1, type=float, help="dropout probability")
parser.add_argument('--forward-passes', default=1, type=int, help="number of MC dropout forward passes for validation")


def set_training_seed(seed):
    # Function to set the different seeds 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def train(model, device, train_loader, optimizer, criterion, epoch, scheduler=None):
    print('\nEpoch: %d' % epoch)
    model.train()
    running_loss = 0
    epoch_loss = 0 # to store total loss for the epoch
    num_batches = len(train_loader)
    correct = 0
    total = 0
    results = {}

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
        epoch_loss += loss.item()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, batch_idx + 1, running_loss / 100)) # print epoch, batch_index and average loss over that mini-batch
            running_loss = 0.0
        
        wandb.log({"training_loss":loss.item(), "lr": optimizer.param_groups[0]['lr'], })
        
    # Store the epoch loss and accuracy
    average_epoch_loss = epoch_loss / num_batches
    train_accuracy = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']

    wandb.log({"epoch": epoch+1, "training_accuracy": train_accuracy, "avg_epoch_loss": average_epoch_loss})

    results["epoch"] = epoch
    results["avg_epoch_loss"] = average_epoch_loss
    results["train_accuracy"] = train_accuracy
    results["lr"] = current_lr

    return results 

def evaluate(model, test_loader, device, epoch=None, num_forward_passes=1, phase="validation"):
    model.eval()
    enable_dropout(model)
    correct = 0
    total = len(test_loader.dataset)
    total_nll = 0.0
    num_classes = 10 # CIFAR-10
    ece_metric = CalibrationError(n_bins=15, norm='l1', task="multiclass", num_classes=num_classes).to(device)

    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)

            # Handle multiple forward passes
            logits = torch.stack([model(inputs) for _ in range(num_forward_passes)], dim=2)
            probs = torch.softmax(logits, dim=1)

            # Duplicate labels for the ensemble size and num_eval_samples
            labels_expanded = labels.unsqueeze(1).expand(-1, num_forward_passes) # Shape: (batch_size, num_forward_passes)

            # Compute the log likelihoods
            log_likelihoods = - F.cross_entropy(logits, labels_expanded, reduction="none") # Shape: (batch_size, num_forward_passes)
            logsumexp_temp = - torch.logsumexp(log_likelihoods, dim=1) + math.log(num_forward_passes) # Eq. 14 in the paper

            # Return the mean NLL across the batch 
            nll = logsumexp_temp.mean() 
            total_nll += nll.item()

            # Average probs over ensemble_size and num_eval_samples, make predictions and compute accuracy
            mean_probs = probs.mean(dim=2) # Shape: (batch_size, num_classes)
            preds = mean_probs.argmax(dim=1) 
            correct += preds.eq(labels).sum().item()

            # Compute ECE
            ece_metric.update(mean_probs, labels)


    average_nll = total_nll / len(test_loader)
    accuracy = 100. * correct / total
    ece = ece_metric.compute().item()

    wandb.log({f"{phase}_nll_loss": average_nll, f"{phase}_accuracy": accuracy, f"{phase}_ece": ece})
    
    if phase == "validation":
        results = {"epoch": epoch, "accuracy": accuracy, "ece": ece, "average_nll": average_nll}
    else:
        results = {"accuracy": accuracy, "ece": ece, "average_nll": average_nll}

    print(f"{phase} Results: {results}")
    return results


def main():

    # Parse arguments
    args = parser.parse_args()
    training_seed = args.seed
    batch_size = args.batch_size
    mode_for_wandb = args.wandb
    if args.use_subset: 
        subset_size = 1000
    else: 
        subset_size = None
    data_seed = 42 # seed used for data loading (e.g. transformations)
    print(f"Are we using a subset? {args.use_subset}")
    
    print(f"Total number of epochs {args.epochs}")

    # Initialize W&B
    if args.use_subset:
        run_name = f"TestRun_LearningRate"
    else:
        run_name = f"B{batch_size}_seed{training_seed}_FP{args.forward_passes}"
    
    # Initialize W&B
   
    wandb.init(project='MC-dropout-WR', mode=mode_for_wandb, name=run_name)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # Data pre-processing
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size, seed=data_seed, subset_size=subset_size)

    # Set seed for training
    set_training_seed(training_seed)

    # Model setup
    model = WideResNet(depth=28, widen_factor=10, num_classes=10, dropRate=args.droprate).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)


    # Learning rate scheduler (if using one, otherwise None)
    print(f"Using the following scheduler: {args.scheduler}")
    if args.scheduler == "warm":
        scheduler = WarmUpPiecewiseConstantSchedule(optimizer=optimizer, steps_per_epoch=len(train_loader), base_lr=args.lr, 
                                                    lr_decay_ratio=0.2, lr_decay_epochs=[60, 120, 160], warmup_epochs=args.warmup_epochs)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
    elif args.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif args.scheduler == "none":
        scheduler = None

    # Results for each epoch
    all_train_results = []
    all_val_results = []
    for epoch in range(args.epochs):
        train_results = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch, scheduler=scheduler)
        val_results = evaluate(model=model, test_loader=val_loader, device=device, epoch=epoch, num_forward_passes=args.forward_passes, phase="validation")
        all_train_results.append(train_results)
        all_val_results.append(val_results)

        if args.scheduler == "multistep":
            scheduler.step()
            print(f'After stepping scheduler, Learning Rate: {optimizer.param_groups[0]["lr"]}')
    
    # Whether to perform evaluation on the testing set
    test_metrics = None
    if args.use_subset is False: 
        test_metrics = evaluate(model=model, test_loader=test_loader, device=device, phase="testing")
        print("Now computing test metrics!")

if __name__ == '__main__':
   main()
    # print("so this works")

