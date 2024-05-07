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

from wide_resnet import WideResNet


# Add parsing functionality 
parser = argparse.ArgumentParser(description='Deterministic Wide ResNet (on CIFAR 10)')

parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='input mini-batch size for training')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=1, type=int, help="seed for reproducibility")
parser.add_argument('--use-scheduler', default=True, type=bool, help="Whether to use a scheduler for the LR or not")
parser.add_argument('--use-subset', default=True, type=bool, help="whether to use a small subset or entire dataset (for debugging locally)")

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

def validate(model, test_loader, device, epoch, metrics=None):
    model.eval()
    total = 0
    total_nll = 0.0
    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Compute log probabilities and calculate NLL loss for the current batch and then accumulate
            log_probs = F.log_softmax(outputs, dim=1) 
            nll_loss = F.nll_loss(log_probs, labels)
            total_nll += nll_loss.item() * inputs.size(0)
            # print(f"the total NLL is {total_nll}")
            
            total += labels.size(0)
            
            
            if metrics:
                for metric in metrics.values():
                    metric.update(outputs, labels)

    # Calculate final metric values
    results = {"epoch": epoch,
               "average_nll": total_nll / total}
    
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
    print(f"The validation results are as follows: {results}")
    return results


def main():
    args = parser.parse_args(args=[])
    
    # Set seed
    torch.manual_seed(args.seed)
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

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # ((m_c1, m_c2, m_c3), (sd_c1, sd_c2, sd_c3))

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Loading the CIFAR-10 dataset 
    train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

    use_subset = args.use_subset
    
    if use_subset: 
        # Creating subset of data to efficiently run model locally 
        # num_samples_train = 500
        num_samples_val = 1000
        # indices_train = np.random.choice(len(train_set), num_samples_train, replace=False)
        indices_val = np.random.choice(len(val_set), num_samples_val, replace=False)

        # subset_train_set = Subset(train_set, indices_train)
        subset_val_set = Subset(val_set, indices_val)

        # train_loader = DataLoader(subset_train_set, batch_size=30, shuffle=True, num_workers=2)
        # val_loader = DataLoader(subset_val_set, batch_size=10, shuffle=False, num_workers=2)
        val_loader = DataLoader(subset_val_set, batch_size=128, shuffle=False, num_workers=2)
    
    # Model setup
    model = WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler (if using one, otherwise None)
    scheduler = None
    if args.use_scheduler:
        print("Now using a scheduler for the LR!!")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)

    # Results for each epoch
    all_results = []
    for epoch in range(args.epochs):
        train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch, scheduler=scheduler)
        results = validate(model=model, test_loader=val_loader, device=device, epoch=epoch, metrics=metrics)
        all_results.append(results)
    
    print(all_results)

    # Save results for later
    current_time = datetime.now().strftime("%m-%d-H%H")
    filename = f'results_{current_time}.json'

    with open(filename, 'w') as file:
        json.dump(all_results, file)

if __name__ == '__main__':
   main()
    # print("so this works")

