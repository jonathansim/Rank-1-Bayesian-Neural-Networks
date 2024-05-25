import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if "weight" in name:
            l2_reg += torch.sum(param ** 2)

    # Total ELBO loss
    total_loss = nll_loss + kl_loss + weight_decay * l2_reg

    return total_loss, nll_loss, kl_loss, kl_div






def truncated_normal(tensor, mean=0.0, stddev=0.1):
    with torch.no_grad():
        tensor.normal_(mean=mean, std=stddev)
        while True:
            cond = torch.logical_or(tensor < mean - 2 * stddev, tensor > mean + 2 * stddev)
            if not cond.any():
                break
            tensor[cond] = tensor[cond].normal_(mean=mean, std=stddev)
    return tensor