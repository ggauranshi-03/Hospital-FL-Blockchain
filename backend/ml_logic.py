import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import medmnist
from medmnist import INFO
import requests
import os
import random
import numpy as np

# --- PINATA CONFIG ---
PINATA_JWT = "" 

def upload_to_pinata(file_path):
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    
    # Use Bearer Token (Standard for Pinata)
    headers = {
        "Authorization": f"Bearer {PINATA_JWT}"
    }
    
    try:
        with open(file_path, "rb") as file:
            files = {"file": file}
            response = requests.post(url, files=files, headers=headers)
            
            if response.status_code == 200:
                # Success!
                return response.json()["IpfsHash"]
            else:
                # Print the exact error from Pinata to your backend terminal
                print(f"❌ PINATA UPLOAD FAILED: {response.status_code}")
                print(f"Error Details: {response.text}")
                return "QmFakeHash_UploadFailed"
                
    except Exception as e:
        print(f"❌ NETWORK ERROR: {e}")
        return "QmFakeHash_UploadFailed"
# Define Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 32 * 13 * 13)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def train_local(num_samples=500,round_id=1):
    """
    Trains the model locally on a random subset of data.
    Returns: (ipfs_hash, accuracy, logs[])
    """
    
    # 1. Load Data
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', download=True, transform=None)
    
    # --- RANDOM SAMPLING (Fixed for Numpy) ---
    total_len = len(train_dataset)
    sample_count = min(num_samples, total_len)
    
    # Randomly select indices
    indices = random.sample(range(total_len), sample_count)
    
    # FIX: Access the underlying numpy array (.imgs) directly
    # train_dataset.imgs is a numpy array (N, 28, 28)
    # train_dataset.labels is a numpy array (N, 1)
    subset_imgs = train_dataset.imgs[indices]
    subset_labels = train_dataset.labels[indices]
    
    # Convert to Tensor [Batch, Channel, Height, Width]
    # We divide by 255.0 to normalize pixel values to 0-1 range
    X = torch.tensor(subset_imgs).float().unsqueeze(1) / 255.0
    y = torch.tensor(subset_labels).long().squeeze()
    
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    # 2. Train (10 Epochs)
    model = Net()
    global_model_path = f"global_model_r{round_id}.pth"
    if os.path.exists(global_model_path):
        print(f"Loading Global Model from Round {round_id}")
        model.load_state_dict(torch.load(global_model_path))
    else:
        print("No global model found, starting from scratch.")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    training_logs = [] 
    
    print(f"Starting training on {sample_count} samples for 10 epochs...")

    for epoch in range(10):
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

        epoch_acc = correct / total
        epoch_loss = total_loss / len(loader)
        
        log_message = f"Epoch {epoch+1}/10 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
        print(log_message)
        training_logs.append(log_message)

    final_accuracy = correct / total
    
    # 3. Save & Upload to Pinata
    model_path = "local_model.pth"
    torch.save(model.state_dict(), model_path)
    
    print("Uploading to Pinata IPFS...")
    ipfs_hash = upload_to_pinata(model_path)
    
    if "QmFake" in ipfs_hash:
        print("⚠️ Upload failed. Check logs above.")
    else:
        print(f"✅ Uploaded! Hash: {ipfs_hash}")
    
    return ipfs_hash, final_accuracy, training_logs