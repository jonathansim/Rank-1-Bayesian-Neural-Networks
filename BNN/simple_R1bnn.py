import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from bnn_conv_layer import Rank1BayesianConv2d
from bnn_linear_layer import Rank1BayesianLinear
from bnn_utils import elbo_loss

class SimpleBNN(nn.Module):
    def __init__(self):
        super(SimpleBNN, self).__init__()
        self.conv1 = Rank1BayesianConv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.fc1 = Rank1BayesianLinear(32*32*32, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = Rank1BayesianLinear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc3 = Rank1BayesianLinear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, 32*32*32)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def kl_divergence(self):
        kl = 0
        for module in self.modules():
            if isinstance(module, (Rank1BayesianConv2d, Rank1BayesianLinear)):
                kl += module.kl_divergence()
        return kl

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, optimizer, and loss function
model = SimpleBNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")


# Training loop
def train(model, optimizer, train_loader, epochs=1, weight_decay=1e-4):
    model.train()
    num_batches = len(train_loader)
    batch_counter = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss, nll_loss, kl_loss = elbo_loss(output, target, model, batch_counter=batch_counter, num_batches=num_batches, kl_annealing_epochs=13, weight_decay=weight_decay)
            loss.backward()
            optimizer.step()

            batch_counter += 1
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, KL: {kl_loss.item():.4f}, NLL: {nll_loss.item():.4f}')
        
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {running_loss / len(train_loader.dataset):.4f}')
    print(f"The batchcounter is at: {batch_counter}")
    print(f"There are this many minibatches: {num_batches}")
# Run the training loop for multiple epochs
train(model, optimizer, train_loader, epochs=16)
