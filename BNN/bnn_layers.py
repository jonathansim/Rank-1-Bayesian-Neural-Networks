'''
Script containing the layers needed for the implementation of a (rank-1) BNN. 
This will initially include a 2D convolutional layer and a dense layer, as the model which'll be utilised is a (wide) resnet. 
'''
import torch
from torch import nn 
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import initializers

class Conv2DRank1(PyroModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: int,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 r_initializer='trainable_normal',
                 s_initializer='trainable_normal',
                 groups=1):
        super(Conv2DRank1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        #self.r_initializer = initializers.get_initializer(r_initializer)
        #self.s_initializer = initializers.get_initializer(s_initializer)

        # Initialize the deterministic weight matrix W
        self.W = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))

        # Apply He initialization (Kaiming)
        torch.nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='relu') # should the mode be fan_out instead? 
        self.W.requires_grad_(False)  # Ensure W is not trainable

        # Rank-1 factorization - define variational parameters for r and s as Pyro samples
        # self.r = PyroSample(dist.Normal(0, 1).expand([out_channels]).to_event(1))
        # self.s = PyroSample(dist.Normal(0, 1).expand([in_channels]).to_event(1))

        self.u = PyroSample(prior=dist.Normal(0., 1.).expand([out_channels, 1, kernel_size, kernel_size]).to_event(4))
        self.v = PyroSample(prior=dist.Normal(0., 1.).expand([1, in_channels, 1, 1]).to_event(4))

        # Define bias if required 
        if bias:
            self.bias = PyroSample(dist.Normal(0, 1).expand([out_channels]).to_event(1))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
         # Rank-1 perturbation - Pyro automatically samples r and s when they are accessed
        rsT = torch.ger(self.r, self.s).view(self.out_channels, self.in_channels, 1, 1)
        perturbed_W = self.W * rsT

        return nn.functional.conv2d(x, perturbed_W, None, self.stride, self.padding) # temporary (currently no prior)
