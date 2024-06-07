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
from bnn_utils import elbo_loss, WarmUpPiecewiseConstantSchedule


# Add parsing functionality 
parser = argparse.ArgumentParser(description='Rank-1 Bayesian Wide ResNet (on CIFAR 10)')

# General arguments
parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='input mini-batch size for training')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=1, type=int, help="seed for reproducibility")
parser.add_argument('--use-scheduler', default=True, type=bool, help="Whether to use a scheduler for the LR or not")
parser.add_argument('--use-subset', default=False, type=bool, help="whether to use a subset (for debugging locally) or all data")
parser.add_argument('--wandb', default="online", type=str, choices=["online", "disabled"] , help="whether to track with weights and biases or not")

# Rank-1 Bayesian specific arguments
parser.add_argument('--ensemble-size', default=1, type=int, help="Number of models in the ensemble")
parser.add_argument('--rank1-distribution', default="normal", type=str, choices=["normal", "cauchy"], help="Rank-1 distribution to use")
parser.add_argument('--prior-mean', default=1.0, type=float, help="Mean for the prior distribution")
parser.add_argument('--prior-stddev', default=0.1, type=float, help="Standard deviation for the prior distribution")
parser.add_argument('--mean-init-std', default=0.5, type=float, help="Standard deviation for the mean initialization")


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
    correct = 0
    total = 0
    num_batches = num_batches
    batch_counter = batch_counter
    num_training_samples = len(train_loader.dataset)

    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        batch_counter += 1
        loss, nll_loss, kl_loss, kl_div = elbo_loss(output, target, model, batch_counter, num_batches, kl_annealing_epochs, num_training_samples, weight_decay)
        loss.backward()

        # Monitor and log gradient norms and afterwards clip gradients to prevent explosion
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)

        optimizer.step()

        # Use LR scheduler 
        # (Below code only if using custom-made WUPCS scheduler or cosine-annealing!! - not MultiStepLR, where it should be put later)
        if scheduler:
            scheduler.step()
            # print(f'After stepping scheduler, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        wandb.log({"loss":loss.item(), "nll_loss": nll_loss.item(), "kl_div": kl_loss.item(), "batch_count": batch_counter, 
                   "kl_div_unscaled": kl_div.item(), "lr": optimizer.param_groups[0]['lr'], "grad_norm": total_norm})
        
    train_accuracy = 100 * correct / total
    # current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"epoch": epoch+1, "accuracy": train_accuracy})


    return batch_counter


def evaluate(model, device, test_loader, epoch=None, phase="Validation"):
    model.eval()
    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)
    total_nll = 0

    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            nll_loss = F.cross_entropy(outputs, labels, reduction="sum")
            total_nll += nll_loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
    accuracy = 100. * correct / total
    average_nll = total_nll / total
    wandb.log({"validation_accuracy": accuracy, "validation_average_nll": average_nll})


def test_evaluate(model, device, test_loader, epoch=None, phase="Testing"):
    model.eval()
    test_loss = 0
    correct = 0
    total = len(test_loader.dataset)
    total_nll = 0

    with torch.no_grad():
        for (inputs, labels) in test_loader: 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            nll_loss = F.cross_entropy(outputs, labels, reduction="sum")
            total_nll += nll_loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / total
    average_nll = total_nll / total

    wandb.log({"testing_accuracy": accuracy, "testing_average_nll": average_nll})
    print(f"Accuracy on the test set: {accuracy}")


def main():

    # Parse arguments
    args = parser.parse_args()
    training_seed = args.seed
    batch_size = args.batch_size // args.ensemble_size # Divide the batch size by the ensemble size (for memory reasons)
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
        run_name = f"run_mixsize_{args.ensemble_size}_128batch_new_scheduler_3normGradClip" 

    wandb.init(project='rank1-bnn-WR', mode=mode_for_wandb, name=run_name)

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
    model = Rank1Bayesian_WideResNet(depth=28, widen_factor=10, num_classes=10, ensemble_size=args.ensemble_size, 
                                     rank1_distribution=args.rank1_distribution, prior_mean=args.prior_mean, 
                                     prior_stddev=args.prior_stddev, mean_init_std=args.mean_init_std).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)

    # Learning rate scheduler (if using one, otherwise None)
    scheduler = None
    if args.use_scheduler:
        print("Now using a scheduler for the LR!!")
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 180], gamma=0.2)
        scheduler = WarmUpPiecewiseConstantSchedule(optimizer=optimizer, steps_per_epoch=len(train_loader), base_lr=args.lr, 
                                                    lr_decay_ratio=0.2, lr_decay_epochs=[80, 160, 180], warmup_epochs=1)
    
    batch_counter = 0
    num_batches = len(train_loader)

    # Choose either or of the below two options
    # kl_annealing_epochs = args.epochs * 2/3 
    kl_annealing_epochs = 200


    # # Training
    for epoch in range(args.epochs):
        batch_counter = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch, 
              batch_counter=batch_counter, num_batches=num_batches, kl_annealing_epochs=kl_annealing_epochs, scheduler=scheduler)
        evaluate(model=model, device=device, test_loader=val_loader)

        # Step the scheduler at the end of each epoch (only if using MultiStepLR, otherwise put in batch loop)
        # if scheduler:
        #     scheduler.step()
        #     print(f'After stepping scheduler, Learning Rate: {optimizer.param_groups[0]["lr"]}')
        
    test_evaluate(model=model, device=device, test_loader=test_loader)

if __name__ == '__main__':
   main()
