import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import INFO

# Define Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128) # Simplified size
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_one_round():
    # 1. Load Data (Simplified: Loads full dataset for demo)
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True, transform=None)
    
    # Take a small slice (simulate local hospital data)
    X = torch.tensor(train_dataset.imgs[:500]).float().unsqueeze(1) / 255.0
    y = torch.tensor(train_dataset.labels[:500]).long().squeeze()
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # 2. Train
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    
    # 3. Save & Hash (Simulated IPFS Hash for demo)
    torch.save(model.state_dict(), "local_model.pth")
    fake_ipfs_hash = "QmHash" + str(int(accuracy*1000)) # Unique mock hash
    
    return fake_ipfs_hash, accuracy