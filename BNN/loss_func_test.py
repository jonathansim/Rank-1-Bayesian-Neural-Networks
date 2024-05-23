import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

torch.manual_seed(0)

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*28*28, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.view(-1, 32*28*28)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = self.fc3(x)
        return x
# Loss function with L2 regularization and optional KL divergence
def elbo_loss(output, target, model, kl_weight=1.0, weight_decay=1e-4):
    # Negative log-likelihood (cross-entropy loss for classification)
    nll_loss = F.cross_entropy(output, target, reduction='sum')
    print(f"The NLL loss is {nll_loss}")
    # KL divergence regularization term (dummy value for this simple model)
    kl_div = 0.0  # No KL divergence in this simple example
    
    # L2 regularization term (weight decay) applied to weights only
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if "weight" in name:
            l2_reg += torch.sum(param ** 2)
            print(f"Adding L2 regularization for parameter: {name}, value: {torch.sum(param ** 2).item()}")
    
    # Total ELBO loss
    total_loss = nll_loss + kl_weight * kl_div + weight_decay * l2_reg
    print("Done computing loss!")
    print(f"The total loss is {total_loss}")
    
    return total_loss

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, optimizer, and loss function
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, optimizer, train_loader, kl_weight=1.0, weight_decay=1e-4):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        output = model(data)
        loss = elbo_loss(output, target, model, kl_weight, weight_decay)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Loss: {loss.item()}')

# Run the training loop
train(model, optimizer, train_loader)


