# Importing packages 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from bnn_conv_layer_mix import Rank1BayesianConv2d
from bnn_linear_layer_mix import Rank1BayesianLinear

class BasicBlock(nn.Module):
    def __init__(self, 
                 in_planes, 
                 out_planes, 
                 stride,
                 rank1_distribution, 
                 ensemble_size, 
                 prior_mean, 
                 prior_stddev, 
                 mean_init_std, 
                 dropRate=0.0):
        
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.1, eps=1e-5)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Rank1BayesianConv2d(in_features=in_planes, out_features=out_planes, kernel_size=3, stride=stride, padding=1,
                                         use_bias=False, rank1_distribution=rank1_distribution, ensemble_size=ensemble_size, 
                                         prior_mean=prior_mean, prior_stddev=prior_stddev, mean_init_std=mean_init_std) 
        
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.1, eps=1e-5)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Rank1BayesianConv2d(in_features=out_planes, out_features=out_planes, kernel_size=3, stride=1, padding=1, 
                                         use_bias=False, rank1_distribution=rank1_distribution, ensemble_size=ensemble_size, 
                                         prior_mean=prior_mean, prior_stddev=prior_stddev, mean_init_std=mean_init_std) 
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)

        # If the number of input channels is not equal to the number of output channels, 
        # then we need to use a 1x1 convolution to change the number of channels
        if not self.equalInOut:
            self.convShortcut = Rank1BayesianConv2d(in_features=in_planes, out_features=out_planes, kernel_size=1, stride=stride, padding=0,
                                                    use_bias=False, rank1_distribution=rank1_distribution, ensemble_size=ensemble_size, 
                                                    prior_mean=prior_mean, prior_stddev=prior_stddev, mean_init_std=mean_init_std)
        else:
            self.convShortcut = None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, rank1_distribution, 
                 ensemble_size, prior_mean, prior_stddev, mean_init_std, dropRate=0.0):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, rank1_distribution, 
                                      ensemble_size, prior_mean, prior_stddev, mean_init_std, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, rank1_distribution, 
                    ensemble_size, prior_mean, prior_stddev, mean_init_std, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(
                in_planes if i == 0 else out_planes, 
                out_planes, 
                stride if i == 0 else 1, 
                rank1_distribution, 
                ensemble_size, 
                prior_mean, 
                prior_stddev, 
                mean_init_std, 
                dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Rank1Bayesian_WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, rank1_distribution='normal', 
                 ensemble_size=4, prior_mean=1.0, prior_stddev=0.1, mean_init_std=0.5, dropRate=0.0):
        super(Rank1Bayesian_WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        # Initial convolution (first_layer param set to True for input duplication)
        self.conv1 = Rank1BayesianConv2d(in_features=3, out_features=nChannels[0], kernel_size=3, stride=1, padding=1,
                                         use_bias=False, rank1_distribution=rank1_distribution, ensemble_size=ensemble_size, 
                                         prior_mean=prior_mean, prior_stddev=prior_stddev, mean_init_std=mean_init_std, first_layer=True)

        # Residual blocks
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, rank1_distribution, ensemble_size, prior_mean, prior_stddev, mean_init_std, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, rank1_distribution, ensemble_size, prior_mean, prior_stddev, mean_init_std, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, rank1_distribution, ensemble_size, prior_mean, prior_stddev, mean_init_std, dropRate)

        # Final layers
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.1, eps=1e-5)
        self.relu = nn.ReLU(inplace=True)
        self.fc = Rank1BayesianLinear(in_features=nChannels[3], out_features=num_classes, rank1_distribution=rank1_distribution, 
                                      ensemble_size=ensemble_size, prior_mean=prior_mean, prior_stddev=prior_stddev, mean_init_std=mean_init_std)
        self.nChannels = nChannels[3]
        self.ensemble_size = ensemble_size
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        batch_size = x.size(0) # Dynamically retrieve the batch size
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)

        # if not self.training:
        #     out = out.view(self.ensemble_size, batch_size, -1)
        #     out = out.permute(1, 2, 0)

        #     return out # TODO Change this to incorporate log mixture likelihood

        return out # For training, return the output directly
        

        # Average over ensemble members
        mean_out = out.view(self.ensemble_size, batch_size, -1).mean(dim=0)
        return mean_out

    def kl_divergence(self):
        kl = 0
        for module in self.modules():
            if isinstance(module, (Rank1BayesianConv2d, Rank1BayesianLinear)):
                kl += module.kl_divergence()
        return kl

