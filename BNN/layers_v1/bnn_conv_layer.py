import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from bnn_utils import he_normal

class Rank1BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bias=False, ensemble_size=1):
        super(Rank1BayesianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Mean weight
        self.weight_mean = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        he_normal(self.weight_mean)  # Apply He normal initialization
        
        # Bias
        if use_bias: 
            self.bias = nn.Parameter(torch.zeros(out_channels))  # Initialize bias to zeros
        else:
            self.register_parameter('bias', None)
        
        # Rank-1 perturbation parameters
        self.u = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1).normal_(mean=1.0, std=0.5))
        self.v = nn.Parameter(torch.Tensor(1, in_channels, kernel_size, kernel_size).normal_(mean=1.0, std=0.5))
        
        # Rank-1 perturbation log-standard-deviation parameters
        self.u_rho = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1).uniform_(-5, -4))  # Initialize with uniform distribution
        self.v_rho = nn.Parameter(torch.Tensor(1, in_channels, kernel_size, kernel_size).uniform_(-3.6, -3.2))   # Initialize with uniform distribution
        
        # Prior distributions
        self.weight_prior = Normal(1.0, 0.1)

    def forward(self, x):
        # Convert rho parameters to standard deviations using softplus
        u_sigma = torch.log1p(torch.exp(self.u_rho)) + 1e-6
        v_sigma = torch.log1p(torch.exp(self.v_rho)) + 1e-6
        
        # Sample perturbations from the Gaussian distributions
        u_sample = Normal(self.u, u_sigma).rsample()
        v_sample = Normal(self.v, v_sigma).rsample()
        
        # Compute the weight matrix with rank-1 perturbation
        weight = (self.weight_mean * u_sample) * v_sample
        
        # Apply the convolutional transformation
        return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)

    def kl_divergence(self):
        # Convert rho parameters to standard deviations using softplus
        u_sigma = torch.log1p(torch.exp(self.u_rho)) + 1e-6
        v_sigma = torch.log1p(torch.exp(self.v_rho)) + 1e-6 
        
        # Create the posterior distributions for u and v
        u_posterior = Normal(self.u, u_sigma)
        v_posterior = Normal(self.v, v_sigma)
        
        # Compute the KL divergence for u and v
        kl_u = kl_divergence(u_posterior, self.weight_prior).sum()
        kl_v = kl_divergence(v_posterior, self.weight_prior).sum()
        
        # Sum the KL divergences
        return kl_u + kl_v
