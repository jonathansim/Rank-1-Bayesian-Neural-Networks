import torch
import torch.nn.functional as F

# Sample input tensor and weight tensor
input = torch.randn(1, 3, 32, 32).to('cuda')  # Move to GPU
weight = torch.randn(6, 3, 5, 5).to('cuda')  # Move to GPU

# Perform convolution
try:
    result = F.conv2d(input, weight, stride=1, padding=0)
    print("Operation successful on GPU. Device:", result.device)
except RuntimeError as e:
    print("RuntimeError on GPU:", e)
    # Optionally, you can fall back to CPU manually
    input = input.to('cpu')
    weight = weight.to('cpu')
    result = F.conv2d(input, weight, stride=1, padding=0)
    print("Fallback to CPU. Device:", result.device)
