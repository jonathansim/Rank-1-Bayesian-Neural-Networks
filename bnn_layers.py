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
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 alpha_initializer='trainable_normal',
                 gamma_initializer='trainable_normal',
                 groups=1):
        super(Conv2DRank1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.alpha_initializer = initializers.get_initializer(alpha_initializer)
        self.gamma_initializer = initializers.get_initializer(gamma_initializer)

        # Rank-1 factorization
        self.alpha = None
        self.gamma = None

        # Define bias if required 
        if bias:
            self.bias = PyroSample(dist.Normal(0, 1).expand([out_channels]).to_event(1))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        