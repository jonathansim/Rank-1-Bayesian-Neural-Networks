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
import wandb

from rank1_wide_resnet import Rank1Bayesian_WideResNet
from data_utils import load_data
from bnn_utils import elbo_loss


# Add parsing functionality 
parser = argparse.ArgumentParser(description='Rank-1 Bayesian Wide ResNet (on CIFAR 10)')

parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='input mini-batch size for training')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=1, type=int, help="seed for reproducibility")
parser.add_argument('--use-scheduler', default=True, type=bool, help="Whether to use a scheduler for the LR or not")
parser.add_argument('--use-subset', default=False, type=bool, help="whether to use a subset (for debugging locally) or all data")

def set_training_seed(seed):
    # Function to set the different seeds 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(model, 
          device, 
          train_loader, 
          optimizer, 
          epoch, 
          batch_counter,
          num_batches,
          weight_decay=1e-4, 
          kl_annealing_epochs = 1, 
          scheduler=None):

    print('\nEpoch: %d' % epoch)
    model.train()
    # running_loss, running_nll, running_kl = 0, 0, 0
    correct = 0
    total = 0
    num_batches = num_batches
    batch_counter = batch_counter

    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        batch_counter += 1
        loss, nll_loss, kl_loss = elbo_loss(output, target, model, batch_counter, num_batches, kl_annealing_epochs, weight_decay)
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        # Use LR scheduler 
        if scheduler:
            scheduler.step()

        # running_loss += loss.item()
        # running_nll += nll_loss.item()
        # running_kl += kl_loss.item()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # if (batch_idx + 1) % 20 == 0:
        #     wandb.log({"loss": running_loss/20, "nll_loss": running_nll/20, "kl_div": running_kl/20})
        #     running_loss, running_nll, running_kl = 0, 0, 0
        wandb.log({"loss":loss.item(), "nll_loss": nll_loss.item(), "kl_div": kl_loss.item(), "batch_count": batch_counter})
        
    train_accuracy = 100 * correct / total
    current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"epoch": epoch+1, "accuracy": train_accuracy, "lr": current_lr})

    return batch_counter


def evaluate(model, device, test_loader, epoch=None, phase="Validation"):
    model.eval()
    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            nll_loss = F.cross_entropy(outputs, labels, reduction="sum")
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            wandb.log({"validation_nll_loss": nll_loss.item()})
    accuracy = 100. * correct / total
    wandb.log({"validation_accuracy": accuracy})


def main():
    # Initialize W&B
    wandb.init(project='rank1-bnn-WR', mode="online")

    args = parser.parse_args()
    training_seed = args.seed
    batch_size = args.batch_size
    if args.use_subset: 
        subset_size = 3000
    else: 
        subset_size = None
    data_seed = 42 # seed used for data loading (e.g. transformations)
    print(f"Are we using a subset? {args.use_subset}")
    
    print(f"Total number of epochs {args.epochs}")
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
    model = Rank1Bayesian_WideResNet(depth=28, widen_factor=10, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Learning rate scheduler (if using one, otherwise None)
    scheduler = None
    if args.use_scheduler:
        print("Now using a scheduler for the LR!!")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
    
    batch_counter = 0
    num_batches = len(train_loader)
    kl_annealing_epochs = args.epochs * 2/3 

    # print(f"Initial u: {model.conv1.u}")
    # print(f"Initial v: {model.conv1.v}")
    # print(f"Initial u_rho: {model.conv1.u_rho}")
    # print(f"Initial v_rho: {model.conv1.v_rho}")

    # # Training
    for epoch in range(args.epochs):
        batch_counter = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch, 
              batch_counter=batch_counter, num_batches=num_batches, kl_annealing_epochs=kl_annealing_epochs, scheduler=scheduler)
        evaluate(model=model, device=device, test_loader=val_loader)
        

if __name__ == '__main__':
   main()
