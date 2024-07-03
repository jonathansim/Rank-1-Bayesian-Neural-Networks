import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributions as dist
import numpy as np


def he_normal(tensor, mode='fan_in', nonlinearity='relu'):
    '''
    This function initializes a tensor of weights with Kaiming/He normal initialization 
    '''
    if isinstance(tensor, nn.Parameter):
        nn.init.kaiming_normal_(tensor, mode=mode, nonlinearity=nonlinearity)
    return tensor


def elbo_loss(output, target, model, batch_counter, num_batches, kl_annealing_epochs, num_data_samples, weight_decay =1e-4):
    '''
    Computes the loss function as given by eq. (2) in Dusenberry et al. (2020). 
    '''
    # Negative log-likelihood
    nll_loss = F.cross_entropy(output, target, reduction='mean')
    
    # KL divergence regularization term 
    kl_div = model.kl_divergence() / num_data_samples
    # print(f"The total KL loss for this batch is: {kl_div}")

    # KL annealing
    kl_scale = batch_counter + 1
    kl_scale /= num_batches * kl_annealing_epochs
    kl_scale = min(1., kl_scale)
    
    kl_loss = kl_div * kl_scale 

    # L2 regularization 
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            l2_reg += torch.sum(param ** 2)

    # Total ELBO loss
    total_loss = nll_loss + kl_loss + weight_decay * l2_reg

    return total_loss, nll_loss, kl_loss, kl_div

class WarmUpPiecewiseConstantSchedule(_LRScheduler):
    '''
    This class implements a learning rate scheduler that combines a warm-up phase with a piecewise constant decay.
    That is, the learning rate is increased linearly from 0 to the base learning rate during the warm-up phase, and then 
    decreased by a factor of decay_ratio at each epoch in decay_epochs (similarly to MultiStepLR).
    Note: can maybe be done better, but needed it done quickly... 
    '''
    def __init__(self, optimizer, steps_per_epoch, base_lr, lr_decay_ratio, lr_decay_epochs, warmup_epochs, last_epoch=-1):
        self.steps_per_epoch = steps_per_epoch
        self.base_lr = base_lr
        self.decay_ratio = lr_decay_ratio
        self.decay_epochs = lr_decay_epochs
        self.warmup_epochs = warmup_epochs
        self.decay_steps = [e * steps_per_epoch for e in lr_decay_epochs]  # Convert epochs to steps
        super(WarmUpPiecewiseConstantSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Calculate the current step
        lr_step = self.last_epoch
        lr_epoch = lr_step / self.steps_per_epoch
        learning_rate = self.base_lr

        # Warm-Up Phase
        if lr_epoch < self.warmup_epochs:
            learning_rate = self.base_lr * lr_step / (self.warmup_epochs * self.steps_per_epoch)
        else:
            # Piecewise Constant Decay Phase
            for i, start_step in enumerate(self.decay_steps):
                if lr_step >= start_step:
                    learning_rate = self.base_lr * (self.decay_ratio ** (i + 1))
                else:
                    break

        return [learning_rate for _ in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



def compute_ece(probs, labels, n_bins=15):
    '''
    Computes the Expected Calibration Error (ECE) of a model. 
    '''
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) * (probs <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].float().mean().item()
            avg_confidence_in_bin = probs[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def kl_divergence_mixture(posterior_means, posterior_stds, prior_mean, prior_std):
    """
    Computes the KL divergence between a mixture of Gaussians posterior and a Gaussian prior using Monte Carlo sampling.
    
    Args:<
        posterior_means (torch.Tensor): Means of the Gaussian components (k, D).
        posterior_stds (torch.Tensor): Standard deviations of the Gaussian components (k, D).
        prior_mean (float): Mean of the Gaussian prior.
        prior_std (float): Standard deviation of the Gaussian prior.
        
    Returns:
        torch.Tensor: The estimated KL divergence.
    """
    k, D = posterior_means.shape
    log_pi = torch.tensor([1.0 / k]).log().to(posterior_means.device)
    
    posterior_dist = dist.Normal(posterior_means, posterior_stds)
    prior_dist = dist.Normal(prior_mean, prior_std)
    samples = posterior_dist.rsample()
    
    # this log_q_w_components differs from the original log_q_w_components by a .permute (1, 0, 2)
    log_q_w_components = log_pi + posterior_dist.log_prob(samples.unsqueeze(1).expand(k, k, D))

    # dim = 1 (instead of the original dim = 0) compensates for difference in log_q_w_components
    log_q_w_sum = torch.logsumexp(log_q_w_components, dim = 1)
    
    log_p_w = prior_dist.log_prob(samples)
    f_w = log_q_w_sum - log_p_w
    kl_div = f_w.mean(dim = 0)

    return kl_div.sum()


def random_sign_initializer(tensor, probs=1.0):
    """
    Initialize the given tensor with +1 or -1 with the given probability.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        probs (float): Probability of +1.
    """
    bernoulli = torch.bernoulli(torch.full(tensor.shape, probs, device=tensor.device, dtype=tensor.dtype))
    tensor.data = 2 * bernoulli - 1