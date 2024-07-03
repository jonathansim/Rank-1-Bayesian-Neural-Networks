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
import json 
from datetime import datetime
import random
import wandb


from rank1_wide_resnet import Rank1Bayesian_WideResNet
from data_utils import load_data
from bnn_utils import elbo_loss, WarmUpPiecewiseConstantSchedule, compute_ece


# Add parsing functionality 
parser = argparse.ArgumentParser(description='Rank-1 Bayesian Wide ResNet (on CIFAR 10)')

# General arguments
parser.add_argument('--epochs', type=int, default=250, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='input mini-batch size for training')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--seed', default=1, type=int, help="seed for reproducibility")
parser.add_argument('--use-subset', default=False, type=bool, help="whether to use a subset (for debugging locally) or all data")
parser.add_argument('--wandb', default="online", type=str, choices=["online", "disabled"] , help="whether to track with weights and biases or not")
parser.add_argument('--scheduler', default="warm", type=str, choices=["warm", "cosine", "multistep", "none"], help="which scheduler to use")
parser.add_argument('--warmup-epochs', default=5, type=int, help="Number of warmup epochs")
parser.add_argument('--optimizer', default="sgd", type=str, choices=["sgd", "adam"], help="which optimizer to use")
parser.add_argument('--save-model', default=True, type=bool, help="whether to save the model or not")

# Rank-1 Bayesian specific arguments
parser.add_argument('--ensemble-size', default=2, type=int, help="Number of models in the ensemble")
parser.add_argument('--rank1-distribution', default="normal", type=str, choices=["normal", "cauchy"], help="Rank-1 distribution to use")
parser.add_argument('--prior-mean', default=0.0, type=float, help="Mean for the prior distribution")
parser.add_argument('--prior-stddev', default=0.1, type=float, help="Standard deviation for the prior distribution")
parser.add_argument('--mean-init-std', default=0.5, type=float, help="Standard deviation for the mean initialization")
parser.add_argument('--num-eval-samples', default=1, type=int, help="Number of samples to use for evaluation")


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
    num_classes = 10 # CIFAR-10
    # ece_metric = CalibrationError(n_bins=15, norm='l1', task="multiclass", num_classes=num_classes).to(device)

    # Initialize accumulators for metrics
    epoch_loss = 0
    epoch_nll_loss = 0
    epoch_kl_loss = 0
    epoch_kl_div = 0
    num_batches_in_epoch = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        batch_counter += 1
        
        # Repeat the target for the ensemble size (to match the output size and utilize vectorized operations)
        target = target.repeat(model.ensemble_size) 
        
        
        loss, nll_loss, kl_loss, kl_div = elbo_loss(output, target, model, batch_counter, num_batches, kl_annealing_epochs, num_training_samples, weight_decay)
        loss.backward()

        # Monitor and log gradient norms and afterwards clip gradients to prevent explosion
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        
        optimizer.step()

        # Use LR scheduler 
        # (Below code only if using custom-made WUPCS scheduler or cosine-annealing!! - not MultiStepLR, where it should be put later)
        if scheduler:
            scheduler.step()
            # print(f'After stepping scheduler, Learning Rate: {optimizer.param_groups[0]["lr"]}')

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Accumulate metrics
        epoch_loss += loss.item()
        epoch_nll_loss += nll_loss.item()
        epoch_kl_loss += kl_loss.item()
        epoch_kl_div += kl_div.item()
        
        wandb.log({"loss":loss.item(), "nll_loss": nll_loss.item(), "kl_div": kl_loss.item(), "batch_count": batch_counter, 
                   "kl_div_unscaled": kl_div.item(), "lr": optimizer.param_groups[0]['lr'], "grad_norm": total_norm})
        
    train_accuracy = 100 * correct / total
    # current_lr = optimizer.param_groups[0]['lr']

    # Calculate average metrics for the epoch
    avg_loss = epoch_loss / num_batches_in_epoch
    avg_nll_loss = epoch_nll_loss / num_batches_in_epoch
    avg_kl_loss = epoch_kl_loss / num_batches_in_epoch
    avg_kl_div = epoch_kl_div / num_batches_in_epoch

    wandb.log({"epoch": epoch+1, "accuracy": train_accuracy, "avg_loss": avg_loss, "avg_nll_loss": avg_nll_loss, 
               "avg_kl_loss": avg_kl_loss, "avg_kl_div": avg_kl_div, "epoch_lr": optimizer.param_groups[0]['lr']})


    return batch_counter


def evaluate(model, device, test_loader, num_eval_samples, epoch=None, phase="validation"):
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
    
    wandb.log({f"{phase}_average_nll": average_nll, f"{phase}_accuracy": accuracy, f"{phase}_ece": ece})

    return accuracy, average_nll, ece


def main():

    # Parse arguments
    args = parser.parse_args()
    training_seed = args.seed
    batch_size = args.batch_size // args.ensemble_size # Divide the batch size by the ensemble size (for memory reasons)
    mode_for_wandb = args.wandb
    if args.use_subset: 
        subset_size = 32*10
    else: 
        subset_size = None
    data_seed = 42 # seed used for data loading (e.g. transformations)
    print(f"Are we using a subset? {args.use_subset}")
    
    print(f"Total number of epochs {args.epochs}")

    # Initialize W&B
    if args.use_subset:
        run_name = f"TestRun_LearningRate"
    else:
        run_name = f"run_M{args.ensemble_size}_B{batch_size}_NumSamp{args.num_eval_samples}_S{args.seed}_PriorStdDev{args.prior_stddev}_NewInit" 
        # M for ensemble size, B for batch size, S for scheduler
    
    wandb.init(project='rank1-bnn-WR', mode=mode_for_wandb, name=run_name)
    
    # Log the arguments to Weights and Biases
    wandb.config.update(args)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")

    # Data pre-processing
    train_loader, val_loader, test_loader = load_data(batch_size=batch_size, seed=data_seed, subset_size=subset_size, do_validation=True)

    # Set seed for training
    set_training_seed(training_seed)

    # Model setup
    model = Rank1Bayesian_WideResNet(depth=28, widen_factor=10, num_classes=10, ensemble_size=args.ensemble_size, 
                                     rank1_distribution=args.rank1_distribution, prior_mean=args.prior_mean, 
                                     prior_stddev=args.prior_stddev, mean_init_std=args.mean_init_std).to(device)
    
    # Optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler (if using one, otherwise None)
    print(f"Using the following scheduler: {args.scheduler}")
    if args.scheduler == "warm":
        scheduler = WarmUpPiecewiseConstantSchedule(optimizer=optimizer, steps_per_epoch=len(train_loader), base_lr=args.lr, 
                                                    lr_decay_ratio=0.2, lr_decay_epochs=[80, 160, 180], warmup_epochs=args.warmup_epochs)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*args.epochs)
    elif args.scheduler == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160, 180], gamma=0.2)
    elif args.scheduler == "none":
        scheduler = None

    batch_counter = 0
    num_batches = len(train_loader)

    # Choose either or of the below two options
    # kl_annealing_epochs = args.epochs * 2/3 
    kl_annealing_epochs = 200


    # # Training
    for epoch in range(args.epochs):
        batch_counter = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, epoch=epoch, 
              batch_counter=batch_counter, num_batches=num_batches, weight_decay=args.weight_decay, kl_annealing_epochs=kl_annealing_epochs, scheduler=scheduler)
        
        # old_evaluate(model=model, device=device, test_loader=val_loader)
        _, _, _ = evaluate(model=model, device=device, test_loader=val_loader, num_eval_samples=args.num_eval_samples, phase="validation")
        print("Done with evaluation")

        if args.scheduler == "multistep":
            scheduler.step()
            print(f'After stepping scheduler, Learning Rate: {optimizer.param_groups[0]["lr"]}')
        
    # Save the model
    if args.save_model:
        seed = args.seed
        ensemble_size = args.ensemble_size
        model_name = f"BNN_seed{seed}_mixture{ensemble_size}_NewInit.pth"
        torch.save(model.state_dict(), model_name)
    
    # Testing
    test_accuracy, test_nll, test_ece = evaluate(model=model, device=device, test_loader=test_loader, num_eval_samples=args.num_eval_samples, phase="testing")
    print(f"Test accuracy: {test_accuracy}, Test NLL: {test_nll}, Test ECE: {test_ece}")
    
    

if __name__ == '__main__':
   main()
