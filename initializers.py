# Should be distribution-based initializers
# My current understanding is that the initializers should return a distribution? It appears they do that in edward2

from pyro.nn import PyroModule, PyroParam
import pyro.distributions as dist 
import torch

# TODO Implement TrainableNormal class
    # TODO Update class such that mean and std. themselves also are initialized by initializers (TruncatedNormal)
# TODO Implement TrainableCauchy class 

class TrainableNormalInitializer(PyroModule):
    def __init__(self, size, mean=0.0, std=1.0):
        super(TrainableNormalInitializer, self).__init__()
        self.mean = PyroParam(torch.full(size, mean))
        self.std = PyroParam(torch.full(size, std), constraint=dist.constraints.positive)

    def forward(self):
        # Return a distribution that can be sampled from or used directly
        return dist.Normal(self.mean, torch.exp(self.std))
    


def get_initializer(initializer: str):
    if initializer == "trainable_normal":
        return TrainableNormalInitializer
    
    elif initializer == "trainable_cauchy":
        pass
