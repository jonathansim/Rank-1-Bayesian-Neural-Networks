import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.classification
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import argparse
import numpy as np
import torchmetrics
import json 
from datetime import datetime
import random
import wandb 

from wide_resnet import WideResNet

from data_utils import load_data
from custom_scheduler import WarmUpPiecewiseConstantSchedule

# Add parsing functionality 
parser = argparse.ArgumentParser(description='Deterministic Wide ResNet (on CIFAR 10)')

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
        
        wandb.log({"loss":loss.item()})
        
    # Store the epoch loss and accuracy
    average_epoch_loss = epoch_loss / num_batches
    train_accuracy = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']

    wandb.log({"epoch": epoch+1, "accuracy": train_accuracy, "lr": current_lr, "avg_epoch_loss": average_epoch_loss})

    results["epoch"] = epoch
    results["avg_epoch_loss"] = average_epoch_loss
    results["train_accuracy"] = train_accuracy
    results["lr"] = current_lr

    print(f"Epoch {epoch} Training Loss: {average_epoch_loss:.3f}, Training Accuracy: {train_accuracy:.2f}%")
    return results 

def evaluate(model, test_loader, device, epoch=None, metrics=None, phase="Validation"):
    model.eval()
    total = 0
    total_nll = 0.0
    results = {}

    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Compute log probabilities and calculate NLL loss for the current batch and then accumulate
            log_probs = F.log_softmax(outputs, dim=1) 
            nll_loss = F.nll_loss(log_probs, labels)
            total_nll += nll_loss.item() * inputs.size(0)
            total += labels.size(0)
               
            if metrics:
                for metric in metrics.values():
                    metric.update(outputs, labels)

    if epoch is not None:
        results["epoch"] = epoch

    # Calculate final metric values
    results["average_nll"] = total_nll / total

    
    # Compute and store results for any additional metrics
    if metrics:
        for name, metric in metrics.items():
            results[name] = metric.compute().item()
            if name == "accuracy":
                results[name] = results[name]*100
            metric.reset()

    # print(f'Accuracy of this network is {results["accuracy"]}')
    # print(f'ECE of this network is {results["ece"]}')
    # print(f"The average NLL of this network is {results["average_nll"]}")

    wandb.log({"validation_nll_loss": results["average_nll"], "validation_accuracy": results["accuracy"], "validation_ece": results["ece"]})
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
    wandb.init(project='deterministic-WR', mode=mode_for_wandb)

    # Set device
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
    metrics = {"accuracy": model_accuracy, "ece": model_ECE} # note that NLL is computed directly in val loop
    
    # Data pre-processing
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size, seed=data_seed, subset_size=subset_size)

    # Set seed for training
    set_training_seed(training_seed)

    # Model setup
    model = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov, weight_decay=args.weight_decay)

    # Learning rate scheduler (if using one, otherwise None)
    scheduler = None
    if args.use_scheduler:
        print("Now using a scheduler for the LR!!")
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
        scheduler = WarmUpPiecewiseConstantSchedule(optimizer=optimizer, steps_per_epoch=len(train_loader), base_lr=args.lr, 
                                                    lr_decay_ratio=0.2, lr_decay_epochs=[60, 120, 160], warmup_epochs=1)

    # Results for each epoch
    all_train_results = []
    all_val_results = []
    for epoch in range(args.epochs):
        train_results = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch, scheduler=scheduler)
        val_results = evaluate(model=model, test_loader=val_loader, device=device, epoch=epoch, metrics=metrics, phase="Validation")
        all_train_results.append(train_results)
        all_val_results.append(val_results)
    
    # Whether to perform evaluation on the testing set
    test_metrics = None
    if args.use_subset is False: 
        test_metrics = evaluate(model=model, test_loader=test_loader, device=device, metrics=metrics, phase="Testing")
        print("Now computing test metrics!")

    # Save results for later
    current_time = datetime.now().strftime("%m-%d-H%H")
    filename_train = f'det_train_results_{current_time}.json'
    filename_val = f'det_val_results_{current_time}.json'
    filename_test = f'det_test_results_{current_time}.json'

    #### UNCOMMENT BELOW TO SAVE RESULTS TO JSON FILES!!! ####

    # with open(filename_train, 'w') as file:
    #     json.dump(all_train_results, file)
    
    # with open(filename_val, 'w') as file:
    #     json.dump(all_val_results, file)
    
    # if test_metrics is not None: 
    #     with open(filename_test, 'w') as file:
    #         json.dump(test_metrics, file)

if __name__ == '__main__':
   main()
    # print("so this works")

