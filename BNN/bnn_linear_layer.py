import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from bnn_utils import he_normal

class Rank1BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Rank1BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Mean weight and bias
        self.weight_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        he_normal(self.weight_mean, mode='fan_in', nonlinearity='linear') # Apply He normal initialization (linear arg because last layer in wr)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))  # Initialize bias to zeros
        
        # Rank-1 perturbation parameters
        self.u = nn.Parameter(torch.Tensor(out_features, 1).uniform_(-0.1, 0.1))
        self.v = nn.Parameter(torch.Tensor(1, in_features).uniform_(-0.1, 0.1))
        
        # Rank-1 perturbation log-standard-deviation parameters
        self.u_rho = nn.Parameter(torch.Tensor(out_features, 1).uniform_(-5, -4)) # TODO implement more advanced init framework
        self.v_rho = nn.Parameter(torch.Tensor(1, in_features).uniform_(-5, -4))
          
        # Prior distributions
        self.weight_prior = Normal(0, 1)

    def forward(self, x):
        # Convert rho parameters to standard deviations using softplus
        u_sigma = torch.log1p(torch.exp(self.u_rho))
        v_sigma = torch.log1p(torch.exp(self.v_rho))
        
        # Sample perturbations from the Gaussian distributions
        u_sample = Normal(self.u, u_sigma).rsample()
        v_sample = Normal(self.v, v_sigma).rsample()
        
        # Compute the weight matrix with rank-1 perturbation
        weight = (self.weight_mean * u_sample) * v_sample
        
        # Apply the linear transformation
        return F.linear(x, weight, self.bias)

    def kl_divergence(self):
        # Convert rho parameters to standard deviations using softplus
        u_sigma = torch.log1p(torch.exp(self.u_rho))
        v_sigma = torch.log1p(torch.exp(self.v_rho))
        
        u_posterior = Normal(self.u, u_sigma)
        v_posterior = Normal(self.v, v_sigma)
        
        kl_u = kl_divergence(u_posterior, self.weight_prior).sum()
        kl_v = kl_divergence(v_posterior, self.weight_prior).sum()
        
        return kl_u + kl_v
