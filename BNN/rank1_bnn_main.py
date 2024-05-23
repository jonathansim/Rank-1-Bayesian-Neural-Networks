import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.classification
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import argparse
import numpy as np
import torchmetrics
import json 
from datetime import datetime
import random

from R1BNN_wide_resnet import Rank1Bayesian_WideResNet

model = Rank1Bayesian_WideResNet(depth=28, widen_factor=10, num_classes=10)
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in Rank-1 Bayesian WR 28-10: {total_params}")