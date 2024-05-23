import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class StandardCNN(nn.Module):
    def __init__(self):
        super(StandardCNN, self).__init__()
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

# Loss function with L2 regularization
def standard_loss(output, target, model, weight_decay=1e-4):
    # Negative log-likelihood (cross-entropy loss for classification)
    nll_loss = F.cross_entropy(output, target, reduction='mean')
    
    # L2 regularization term (weight decay) applied to weights only
    l2_reg = 0.0
    for name, param in model.named_parameters():
        if "weight" in name:
            l2_reg += torch.sum(param ** 2)
    
    # Total loss
    total_loss = nll_loss + weight_decay * l2_reg
    
    return total_loss

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model, optimizer, and loss function
model = StandardCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {total_params}")

# Training loop
def train(model, optimizer, train_loader, epochs=10, weight_decay=1e-4):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = standard_loss(output, target, model, weight_decay)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f'Epoch [{epoch + 1}/{epochs}], Average Loss: {running_loss / len(train_loader.dataset):.4f}')

# Run the training loop for multiple epochs
train(model, optimizer, train_loader, epochs=10)
