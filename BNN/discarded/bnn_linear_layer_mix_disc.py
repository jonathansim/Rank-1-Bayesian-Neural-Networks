import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence, Cauchy


class Rank1BayesianLinear(nn.Module):
    '''
    A rank-1 Bayesian linear layer. 
    Parameters:
    - rank1_posterior: posterior distribution for rank-1 perturbation parameters
    - ensemble_size: number of ensemble members
    - prior_mean: mean of the prior distribution
    - prior_stddev: standard deviation of the prior distribution
    - dropout_rate_init: dropout rate used for initialization of rank-1 perturbation standard deviations
    - mean_init_std: standard deviation of the normal distribution used for initialization of the mean of the rank-1 perturbation parameters 
                     - should be opposite sign of what used in paper (since they simply put a minus in front themselves)
    '''
    def __init__(self, in_features, 
                 out_features, 
                 rank1_posterior="normal", 
                 ensemble_size = 1, 
                 prior_mean=1.0, 
                 prior_stddev=0.1, 
                 dropout_rate_init=0.001,
                 mean_init_std=0.5):
        
        super(Rank1BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.rank1_posterior = rank1_posterior
        self.prior_mean = prior_mean
        self.prior_stddev = prior_stddev
        self.mean_init_std = mean_init_std
        self.dropout_rate_init = dropout_rate_init
        
        # Shared weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)) 
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # Rank-1 perturbation parameters
        self.u = nn.Parameter(torch.Tensor(ensemble_size, out_features, 1))
        self.v = nn.Parameter(torch.Tensor(ensemble_size, 1, in_features))
        
        # Rank-1 perturbation log-standard-deviation parameters
        self.u_rho = nn.Parameter(torch.Tensor(ensemble_size, out_features, 1)) 
        self.v_rho = nn.Parameter(torch.Tensor(ensemble_size, 1, in_features))
          
        # Prior distributions
        if self.rank1_posterior == "normal":
            self.weight_prior = Normal(prior_mean, prior_stddev)
        elif self.rank1_posterior == "cauchy":
            self.weight_prior = Cauchy(prior_mean, prior_stddev)
        

        self.initialize_parameters() # Initialize all parameters

    def initialize_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear') # Apply He normal initialization (linear arg because last layer in wr)
        nn.init.zeros_(self.bias) # Initialize bias to zeros
        nn.init.normal_(self.u, mean=1.0, std=self.mean_init_std) # Initialize rank-1 perturbation parameters
        nn.init.normal_(self.v, mean=1.0, std=self.mean_init_std) # Initialize rank-1 perturbation parameters

        stddev_init = np.log(np.expm1(np.sqrt(self.dropout_rate_init / (1. - self.dropout_rate_init))))
        nn.init.trunc_normal_(self.u_rho, mean=stddev_init, std=0.1, a=stddev_init-2*0.1, b=stddev_init+2*0.1) # Initialize rank-1 log-std dev parameters
        nn.init.trunc_normal_(self.v_rho, mean=stddev_init, std=0.1, a=stddev_init-2*0.1, b=stddev_init+2*0.1) # Initialize rank-1 log-std dev parameters
        

    def forward(self, x):
        print(f"The shape of the input in the beginning is {x.shape}")
        if x.dim() == 3:
            batch_size = x.size(1)
        else:
            batch_size = x.size(0)
            x = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)
    
        # Convert rho parameters to standard deviations using softplus
        u_sigma = torch.log1p(torch.exp(self.u_rho)) + 1e-6
        v_sigma = torch.log1p(torch.exp(self.v_rho)) + 1e-6
        
        # Sample perturbations from the Gaussian distributions
        u_sample = Normal(self.u, u_sigma).rsample()
        v_sample = Normal(self.v, v_sigma).rsample()
        
        # Compute the weight matrix with rank-1 perturbation
        perturbed_weight = (self.weight.unsqueeze(0) * u_sample) * v_sample

        # Reshape for batch processing
        # x_reshaped = x.contiguous().view(-1, self.in_features)  # (ensemble_size * batch_size, in_features)
        # perturbed_weight_reshaped = perturbed_weight.view(-1, self.in_features)  # (ensemble_size * out_features, in_features)
        # bias_reshaped = self.bias.unsqueeze(0).expand(self.ensemble_size, -1).contiguous().view(-1) # (ensemble_size * out_features)
        
        x_reshaped = x.contiguous().view(self.ensemble_size * batch_size, self.in_features)  # (ensemble_size * batch_size, in_features)
        perturbed_weight_reshaped = perturbed_weight.view(self.ensemble_size * self.out_features, self.in_features)  # (ensemble_size * out_features, in_features)
        # bias_reshaped = self.bias.unsqueeze(0).expand(self.ensemble_size, -1).contiguous().view(self.ensemble_size * self.out_features) # (ensemble_size * out_features)

        print(f"The shape of the perturbed weight is {perturbed_weight.shape}")
        print(f"The shape of the input is {x.shape}")
        print(f"The shape of the reshaped input is {x_reshaped.shape}")
        # Apply the linear transformation
        output = F.linear(x_reshaped, perturbed_weight_reshaped, self.bias.repeat(self.ensemble_size))
        print(f"The shape of the output is {output.shape}")
        output_reshaped = output.view(self.ensemble_size, batch_size, self.out_features)

        return output_reshaped

    def kl_divergence(self):
        # Convert rho parameters to standard deviations using softplus
        u_sigma = torch.log1p(torch.exp(self.u_rho)) + 1e-6
        v_sigma = torch.log1p(torch.exp(self.v_rho)) + 1e-6
        
        u_posterior = Normal(self.u, u_sigma)
        v_posterior = Normal(self.v, v_sigma)
        
        kl_u = kl_divergence(u_posterior, self.weight_prior).sum() # Note to self: Maybe divide by ensemble_size?
        kl_v = kl_divergence(v_posterior, self.weight_prior).sum()
        
        return kl_u + kl_v
