import torch
import torch.nn as nn
import numpy as np

from bnn_linear_layer_mix import Rank1BayesianLinear

def test_correct_computation():
    num_models = 2
    layer = Rank1BayesianLinear(3, 4, ensemble_size=num_models, first_layer=True)
    layer.weight
    input_data = torch.randn(2, 3)  # Batch size of 32
    print(input_data)
    output = layer(input_data)

    assert output.shape == (2*num_models, 4), "Output shape is incorrect"
    print(output)



# Run the tests
if __name__ == "__main__":
    test_correct_computation()