import torch.nn as nn
from wide_resnet import WideResNet


# Check number of parameters in deterministic wide_resnet 28-10
model = WideResNet(depth=28, widen_factor=10, num_classes=10)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in WR 28-10: {total_params}")

