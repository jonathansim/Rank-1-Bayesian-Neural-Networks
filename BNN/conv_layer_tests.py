import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from bnn_conv_layer_mix import Rank1BayesianConv2d
from bnn_linear_layer_mix import Rank1BayesianLinear

class BayesianNN(nn.Module):
    def __init__(self, ensemble_size=1):
        super(BayesianNN, self).__init__()
        self.ensemble_size = ensemble_size
        self.conv1 = Rank1BayesianConv2d(3, 16, kernel_size=3, stride=1, padding=1, ensemble_size=ensemble_size, first_layer=True)
        self.conv2 = Rank1BayesianConv2d(16, 32, kernel_size=3, stride=1, padding=1, ensemble_size=ensemble_size)
        self.fc = Rank1BayesianLinear(32 * 8 * 8, 10, ensemble_size=ensemble_size)

    def forward(self, x):
        print(f"The shape of the input is {x.shape}")
        x = self.conv1(x)
        print(f"The shape of the input after the first convolution is {x.shape}")
        x = F.relu(F.max_pool2d(x, 2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, 2))
        print(f"The shape of the input before flattening is {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"The shape of the input after flattening is {x.shape}")
        x = self.fc(x)
        return x

    def kl_divergence(self):
        kl_div = self.conv1.kl_divergence() + self.conv2.kl_divergence() + self.fc.kl_divergence()
        return kl_div


def test_bayesian_nn():
    num_models = 8
    model = BayesianNN(ensemble_size=num_models)
    input_data = torch.randn(32, 3, 32, 32)  # Batch size of 32, 3 color channels, 32x32 image
    output = model(input_data)
    
    # Check output shape
    assert output.shape == (32*num_models, 10), f"Output shape is incorrect, got {output.shape}"
    print("Output shape test passed")

    # Check if each ensemble member's output is different
    batch_size_per_ensemble = 32
    different_outputs = any(not torch.equal(output[i*batch_size_per_ensemble:(i+1)*batch_size_per_ensemble], 
                                             output[j*batch_size_per_ensemble:(j+1)*batch_size_per_ensemble]) 
                            for i in range(model.ensemble_size) 
                            for j in range(i + 1, model.ensemble_size))
    
    if num_models == 1:
        assert not different_outputs, "Ensemble members should produce the same output"
    else:
        assert different_outputs, "Ensemble members should produce different outputs"
    
    print("Ensemble independence test passed")

    # Check KL divergence
    kl_div = model.kl_divergence()
    assert kl_div > 0, "KL divergence should be positive"
    print("KL divergence test passed")

def run_tests():
    test_bayesian_nn()
    print("All tests passed")

# Run the tests
run_tests()
