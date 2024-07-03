import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, kl_divergence, Cauchy
from bnn_utils import kl_divergence_mixture, random_sign_initializer


class Rank1BayesianConv2d(nn.Module):
    '''
    A rank-1 Bayesian Conv2D layer. 
    Parameters:
    - rank1_distribution: distribution family for rank-1 perturbation parameters
    - ensemble_size: number of ensemble members
    - prior_mean: mean of the prior distribution
    - prior_stddev: standard deviation of the prior distribution
    - dropout_rate_init: dropout rate used for initialization of rank-1 perturbation standard deviations
    - mean_init_std: standard deviation of the normal distribution used for initialization of the mean of the rank-1 perturbation parameters 
                     - should be opposite sign of what used in paper (since they simply put a minus in front themselves)
    - first_layer: boolean indicating if this is the first layer in the network (used for input duplication)
    '''
    def __init__(self, in_features, 
                 out_features,
                 kernel_size,
                 stride=1,
                 padding=0,
                 use_bias=False,                
                 rank1_distribution="normal", 
                 ensemble_size = 1, 
                 prior_mean=1.0, 
                 prior_stddev=0.1, 
                 mean_init_std=0.5,
                 first_layer=False):
        
        super(Rank1BayesianConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.ensemble_size = ensemble_size
        self.rank1_distribution = rank1_distribution
        self.prior_mean = prior_mean
        self.prior_stddev = prior_stddev
        self.mean_init_std = mean_init_std
        self.first_layer = first_layer
        
        
    
        # Shared weight (created directly in layer) and bias    
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride=stride, padding=padding, bias=False)

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        
        # Rank-1 perturbation parameters
        self.r = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        self.s = nn.Parameter(torch.Tensor(ensemble_size, in_features))
        
        # Rank-1 perturbation log-standard-deviation parameters
        self.r_rho = nn.Parameter(torch.Tensor(ensemble_size, out_features)) 
        self.s_rho = nn.Parameter(torch.Tensor(ensemble_size, in_features))
          
        # Prior distributions and dropout rate initialization
        if self.rank1_distribution == "normal":
            self.weight_prior = Normal(prior_mean, prior_stddev)
            self.dropout_rate_init = 0.001
    
        elif self.rank1_distribution == "cauchy":
            self.weight_prior = Cauchy(prior_mean, prior_stddev)
            self.dropout_rate_init = 10**-6
        
        self.initialize_parameters() # Initialize all parameters

    def initialize_parameters(self):
        # Initialize weight and bias with He normal initialization and zeros, respectively
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu') 
        if self.use_bias:
            nn.init.zeros_(self.bias)
        
        # Initialize rank-1 perturbation parameters (mean)
        if self.mean_init_std > 0:
            random_sign_initializer(self.r, probs=self.mean_init_std)
            random_sign_initializer(self.s, probs=self.mean_init_std)
        else:
            nn.init.normal_(self.r, mean=1.0, std=-self.mean_init_std) 
            nn.init.normal_(self.s, mean=1.0, std=-self.mean_init_std) 

        # Initialize rank-1 log-std dev parameters
        stddev_init = np.log(np.expm1(np.sqrt(self.dropout_rate_init / (1. - self.dropout_rate_init))))
        # stddev_init = np.sqrt(self.dropout_rate_init / (1 - self.dropout_rate_init))

        if self.rank1_distribution == "normal":
            nn.init.trunc_normal_(self.r_rho, mean=stddev_init, std=0.1, a=stddev_init-2*0.1, b=stddev_init+2*0.1) 
            nn.init.trunc_normal_(self.s_rho, mean=stddev_init, std=0.1, a=stddev_init-2*0.1, b=stddev_init+2*0.1)
        elif self.rank1_distribution == "cauchy":
            nn.init.constant_(self.r_rho, stddev_init)
            nn.init.constant_(self.s_rho, stddev_init)
        
        

    def forward(self, x):
        # print(f"The shape of the input in the beginning is {x.shape}")
        if self.first_layer:
            # Repeat the input for each ensemble member
            # print("First layer check activated")
            # x = torch.cat([x for i in range(self.ensemble_size)], dim=0)
            x = x.repeat(self.ensemble_size, 1, 1, 1)
            
        # print(f"The shape of the input after the 'first layer' check is {x.shape}")

        num_examples_per_ensemble = x.size(0) // self.ensemble_size
        

        # Convert rho parameters to standard deviations using softplus
        r_sigma = torch.log1p(torch.exp(self.r_rho)) + 1e-6
        s_sigma = torch.log1p(torch.exp(self.s_rho)) + 1e-6
        
         # Sample perturbations from the Gaussian or Cauchy distributions
        if self.rank1_distribution == "normal":
            # r_sample = Normal(self.r, r_sigma).rsample()
            # s_sample = Normal(self.s, s_sigma).rsample()
            r_sample = Normal(self.r, r_sigma).rsample((num_examples_per_ensemble,))
            s_sample = Normal(self.s, s_sigma).rsample((num_examples_per_ensemble,))

        elif self.rank1_distribution == "cauchy":
            r_sample = Cauchy(self.r, r_sigma).rsample()
            s_sample = Cauchy(self.s, s_sigma).rsample()
        
        R = r_sample.permute(1, 0, 2).contiguous().view(-1, self.out_features).unsqueeze_(-1).unsqueeze_(-1)
        S = s_sample.permute(1, 0, 2).contiguous().view(-1, self.in_features).unsqueeze_(-1).unsqueeze_(-1)
        # R = r_sample.repeat(1, num_examples_per_ensemble).view(-1, self.out_features).unsqueeze_(-1).unsqueeze_(-1)
        # S = s_sample.repeat(1, num_examples_per_ensemble).view(-1, self.in_features).unsqueeze_(-1).unsqueeze_(-1)
        
        if self.bias is not None:
            bias = self.bias.repeat(1, num_examples_per_ensemble).view(-1, self.out_features)
            bias.unsqueeze_(-1).unsqueeze_(-1)


        # Apply linear transformation and add bias
        result = self.conv(x * S) * R
        
        if self.bias is not None:
            result += bias
    
        return result

    def kl_divergence(self):
        # Convert rho parameters to standard deviations using softplus
        r_sigma = torch.log1p(torch.exp(self.r_rho)) + 1e-6
        s_sigma = torch.log1p(torch.exp(self.s_rho)) + 1e-6
        
        # # Sample perturbations from the Gaussian or Cauchy distributions
        # if self.rank1_distribution == "normal":
        #     r_posterior = Normal(self.r, r_sigma)
        #     s_posterior = Normal(self.s, s_sigma)
        # elif self.rank1_distribution == "cauchy":
        #     r_posterior = Cauchy(self.r, r_sigma)
        #     s_posterior = Cauchy(self.s, s_sigma)
        
        # # Compute KL divergence between the posteriors and the prior, sum and divide by ensemble size
        # kl_r = kl_divergence(r_posterior, self.weight_prior).sum() / self.ensemble_size
        # kl_s = kl_divergence(s_posterior, self.weight_prior).sum() / self.ensemble_size

        kl_r = kl_divergence_mixture(self.r, r_sigma, self.prior_mean, self.prior_stddev)
        kl_s = kl_divergence_mixture(self.s, s_sigma, self.prior_mean, self.prior_stddev)
        
        return kl_r + kl_s
