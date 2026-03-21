# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, Subset
# import medmnist
# from medmnist import INFO
# import requests
# import os
# import random
# import numpy as np

# # --- PINATA CONFIG ---
# PINATA_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiJlNjY4ZWE5Zi1kZDA4LTRjOWEtOTg0OC0zZjdjMDNiNjEyYmEiLCJlbWFpbCI6ImdhdXJhbnNoaWd1cHRhMjAwMEBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicGluX3BvbGljeSI6eyJyZWdpb25zIjpbeyJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MSwiaWQiOiJGUkExIn0seyJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MSwiaWQiOiJOWUMxIn1dLCJ2ZXJzaW9uIjoxfSwibWZhX2VuYWJsZWQiOmZhbHNlLCJzdGF0dXMiOiJBQ1RJVkUifSwiYXV0aGVudGljYXRpb25UeXBlIjoic2NvcGVkS2V5Iiwic2NvcGVkS2V5S2V5IjoiYzM1ZjU2YWQ0OWI3ZTUzM2Y4OTUiLCJzY29wZWRLZXlTZWNyZXQiOiJhYzhhMzBjZTMxMzU1MGUzZDBlMmY2YTRhOWJjZDU4N2U5NDBkMTZlNWEwZWI2ODUyY2E2Y2RlY2FjMDBlMzVkIiwiZXhwIjoxODAwMjcyNjM3fQ.EHfI6CSy-z0G8KbOS_ZPbLIoHUDUcYVo6wsWvNuBAYQ" 


# def get_hash_by_filename(filename):
#     """
#     Uses Pinata API to search for a CID based on the filename.
#     This ensures we always find the correct global baseline without 
#     needing to store the hash on the blockchain.
#     """
#     url = f"https://api.pinata.cloud/data/pinList?metadata[name]={filename}&status=pinned"
#     headers = {"Authorization": f"Bearer {PINATA_JWT}"}
    
#     try:
#         response = requests.get(url, headers=headers)
#         if response.status_code == 200:
#             data = response.json()
#             if data["rows"]:
#                 # Returns the most recent CID for this filename
#                 return data["rows"][0]["ipfs_pin_hash"]
#         print(f"⚠️ Pinata Search: Could not find CID for filename: {filename}")
#         return None
#     except Exception as e:
#         print(f"❌ Pinata Search Error: {e}")
#         return None

# def fetch_global_model_from_ipfs(round_id, ipfs_hash):
#     """
#     Downloads the .pth file from IPFS and saves it locally.
#     Overwrites any existing file to ensure the model is fresh.
#     """
#     url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
#     save_path = f"global_model_r{round_id}.pth"
    
#     print(f"📡 MANDATORY SYNC: Downloading Global Model for Round {round_id}...")
    
#     try:
#         response = requests.get(url, timeout=60)
#         if response.status_code == 200:
#             with open(save_path, "wb") as f:
#                 f.write(response.content)
#             print(f"✅ Successfully synced {save_path} from IPFS.")
#             return True
#         else:
#             print(f"❌ IPFS Download Failed: {response.status_code}")
#             return False
#     except Exception as e:
#         print(f"❌ Network Error during IPFS fetch: {e}")
#         return False

# def upload_to_pinata(file_path):
#     """
#     Uploads the local model update back to Pinata IPFS.
#     """
#     url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
#     headers = {"Authorization": f"Bearer {PINATA_JWT}"}
    
#     try:
#         with open(file_path, "rb") as file:
#             files = {"file": file}
#             response = requests.post(url, files=files, headers=headers)
#             if response.status_code == 200:
#                 return response.json()["IpfsHash"]
#             else:
#                 print(f"❌ PINATA UPLOAD FAILED: {response.status_code}")
#                 return "QmFakeHash_UploadFailed"
#     except Exception as e:
#         print(f"❌ NETWORK ERROR: {e}")
#         return "QmFakeHash_UploadFailed"

# # --- MODEL ARCHITECTURE ---
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
#         self.relu = nn.ReLU()
#         self.pool = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(32 * 13 * 13, 128)
#         self.fc2 = nn.Linear(128, 2)

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = x.view(-1, 32 * 13 * 13)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# def train_local(num_samples=500, round_id=1):
#     """
#     The automated training loop:
#     1. Finds global model CID by name on Pinata.
#     2. Downloads weights from IPFS.
#     3. Loads weights into Net and trains for 10 epochs.
#     4. Uploads result back to Pinata.
#     """
    
#     # 1. Load Local Data (PneumoniaMNIST)
#     info = INFO['pneumoniamnist']
#     DataClass = getattr(medmnist, info['python_class'])
#     train_dataset = DataClass(split='train', download=True, transform=None)
    
#     total_len = len(train_dataset)
#     sample_count = min(num_samples, total_len)
#     indices = random.sample(range(total_len), sample_count)
    
#     subset_imgs = train_dataset.imgs[indices]
#     subset_labels = train_dataset.labels[indices]
    
#     X = torch.tensor(subset_imgs).float().unsqueeze(1) / 255.0
#     y = torch.tensor(subset_labels).long().squeeze()
#     loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

#     # 2. MANDATORY IPFS SYNC
#     model = Net()
#     target_filename = f"global_model_r{round_id}.pth"
    
#     # Search for the CID using the filename
#     global_hash = get_hash_by_filename(target_filename)
    
#     if global_hash:
#         # Mandatory download from IPFS
#         fetch_global_model_from_ipfs(round_id, global_hash)
    
#     # Load weights if the file was successfully downloaded
#     if os.path.exists(target_filename):
#         print(f"✅ Initializing training with Global Baseline: {target_filename}")
#         model.load_state_dict(torch.load(target_filename))
#     else:
#         print(f"⚠️ Baseline {target_filename} not found on IPFS. Starting from scratch.")

#     # 3. Local Training (10 Epochs)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()
#     model.train()
#     training_logs = [] 

#     print(f"Starting training on {sample_count} samples...")

#     for epoch in range(10):
#         total_loss, correct, total = 0, 0, 0
#         for images, labels in loader:
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#         epoch_acc = correct / total
#         epoch_loss = total_loss / len(loader)
        
#         log_message = f"Epoch {epoch+1}/10 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
#         print(log_message)
#         training_logs.append(log_message)

#     final_accuracy = correct / total
    
#     # 4. Save & Upload
#     model_path = "local_model.pth"
#     torch.save(model.state_dict(), model_path)
    
#     print("Uploading local update to Pinata IPFS...")
#     ipfs_hash = upload_to_pinata(model_path)
    
#     return ipfs_hash, final_accuracy, training_logs









import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from torchvision import datasets, transforms
import medmnist
from medmnist import INFO
import requests
import os
import random
import numpy as np

# --- PINATA CONFIG ---
PINATA_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiJlNjY4ZWE5Zi1kZDA4LTRjOWEtOTg0OC0zZjdjMDNiNjEyYmEiLCJlbWFpbCI6ImdhdXJhbnNoaWd1cHRhMjAwMEBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwicGluX3BvbGljeSI6eyJyZWdpb25zIjpbeyJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MSwiaWQiOiJGUkExIn0seyJkZXNpcmVkUmVwbGljYXRpb25Db3VudCI6MSwiaWQiOiJOWUMxIn1dLCJ2ZXJzaW9uIjoxfSwibWZhX2VuYWJsZWQiOmZhbHNlLCJzdGF0dXMiOiJBQ1RJVkUifSwiYXV0aGVudGljYXRpb25UeXBlIjoic2NvcGVkS2V5Iiwic2NvcGVkS2V5S2V5IjoiYzM1ZjU2YWQ0OWI3ZTUzM2Y4OTUiLCJzY29wZWRLZXlTZWNyZXQiOiJhYzhhMzBjZTMxMzU1MGUzZDBlMmY2YTRhOWJjZDU4N2U5NDBkMTZlNWEwZWI2ODUyY2E2Y2RlY2FjMDBlMzVkIiwiZXhwIjoxODAwMjcyNjM3fQ.EHfI6CSy-z0G8KbOS_ZPbLIoHUDUcYVo6wsWvNuBAYQ" 

def get_hash_by_filename(filename):
    url = f"https://api.pinata.cloud/data/pinList?metadata[name]={filename}&status=pinned"
    headers = {"Authorization": f"Bearer {PINATA_JWT}"}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data["rows"]:
                return data["rows"][0]["ipfs_pin_hash"]
        return None
    except Exception as e:
        print(f"❌ Pinata Search Error: {e}")
        return None

def fetch_global_model_from_ipfs(round_id, ipfs_hash):
    url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
    save_path = f"global_model_r{round_id}.pth"
    print(f"📡 MANDATORY SYNC: Downloading Global Model for Round {round_id}...")
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Successfully synced {save_path} from IPFS.")
            return True
        else:
            print(f"❌ IPFS Download Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Network Error during IPFS fetch: {e}")
        return False

def upload_to_pinata(file_path):
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {"Authorization": f"Bearer {PINATA_JWT}"}
    try:
        with open(file_path, "rb") as file:
            files = {"file": file}
            response = requests.post(url, files=files, headers=headers)
            if response.status_code == 200:
                return response.json()["IpfsHash"]
            return "QmFakeHash_UploadFailed"
    except Exception as e:
        print(f"❌ NETWORK ERROR: {e}")
        return "QmFakeHash_UploadFailed"

# --- MODEL ARCHITECTURE ---
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

def train_local(num_samples=500, round_id=1, data_path=None):
    """
    Trains on either MedMNIST (default) or Custom Data (if data_path is provided).
    """
    
    # 1. LOAD DATA (CUSTOM OR DEFAULT)
    if data_path and os.path.exists(data_path):
        print(f"📂 LOADING CUSTOM DATASET from: {data_path}")
        
        # Transform: Resize to 28x28, Grayscale, Tensor, Normalize
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load Folder Structure (e.g. data/Normal, data/Pneumonia)
        try:
            full_dataset = datasets.ImageFolder(root=data_path, transform=data_transform)
            
            # Simple split: 80% train, 20% ignored/val (for this demo we just train)
            # If dataset is small, use all of it.
            train_dataset = full_dataset
            print(f"✅ Loaded {len(train_dataset)} custom images successfully.")
            
        except Exception as e:
            print(f"❌ Error loading custom data: {e}. Fallback to MedMNIST.")
            data_path = None # Trigger fallback

    if not data_path:
        # Fallback to MedMNIST
        print("⚠️ Using Default MedMNIST (Pneumonia)...")
        info = INFO['pneumoniamnist']
        DataClass = getattr(medmnist, info['python_class'])
        full_dataset = DataClass(split='train', download=True, transform=transforms.ToTensor())
        
        total_len = len(full_dataset)
        sample_count = min(num_samples, total_len)
        indices = random.sample(range(total_len), sample_count)
        
        # Manual subset creation for MedMNIST arrays
        subset_imgs = full_dataset.imgs[indices]
        subset_labels = full_dataset.labels[indices]
        X = torch.tensor(subset_imgs).float().unsqueeze(1) / 255.0
        y = torch.tensor(subset_labels).long().squeeze()
        train_dataset = TensorDataset(X, y)

    # Create Loader
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 2. MANDATORY IPFS SYNC
    model = Net()
    target_filename = f"global_model_r{round_id}.pth"
    global_hash = get_hash_by_filename(target_filename)
    
    if global_hash:
        fetch_global_model_from_ipfs(round_id, global_hash)
    
    if os.path.exists(target_filename):
        print(f"✅ Initializing with Global Baseline: {target_filename}")
        model.load_state_dict(torch.load(target_filename))
    else:
        print(f"⚠️ Baseline {target_filename} not found. Starting fresh.")

    # 3. Local Training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    training_logs = [] 

    print("Starting training loop...")
    for epoch in range(10):
        total_loss, correct, total = 0, 0, 0
        for batch in loader:
            # Handle difference between ImageFolder (returns x, y) and TensorDataset
            images, labels = batch
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_acc = correct / total if total > 0 else 0
        epoch_loss = total_loss / len(loader) if len(loader) > 0 else 0
        
        log_message = f"Epoch {epoch+1}/10 - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}"
        print(log_message)
        training_logs.append(log_message)

    final_accuracy = correct / total if total > 0 else 0
    
    # 4. Save & Upload
    model_path = "local_model.pth"
    torch.save(model.state_dict(), model_path)
    print("Uploading local update to Pinata IPFS...")
    ipfs_hash = upload_to_pinata(model_path)
    
    return ipfs_hash, final_accuracy, training_logs