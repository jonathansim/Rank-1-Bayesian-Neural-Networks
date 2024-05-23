# Importing packages 
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Basic block structure
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential() # initializes empty seq. model - This setup is used when no modification is needed on the shortcut connection;
        # that is, when the input dimension matches the output dimension and no downsampling is needed.
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Wide ResNet class
class WideResNet(nn.Module):
    def __init__(self, block=BasicBlock, depth=28, widen_factor=10, num_classes=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), "Depth should be 6n+4"
        n = (depth - 4) // 6

        # k = widen_factor
        num_blocks = [n, n, n]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16*widen_factor, num_blocks[0], stride=1, dropout_rate=dropRate)
        self.layer2 = self._make_layer(block, 32*widen_factor, num_blocks[1], stride=2, dropout_rate=dropRate)
        self.layer3 = self._make_layer(block, 64*widen_factor, num_blocks[2], stride=2, dropout_rate=dropRate)
        self.linear = nn.Linear(64*widen_factor*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        # out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
