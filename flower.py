import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset

# --- 1. THE ML MODEL & TRAINING LOOP ---
class MedicalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2) # Example: Binary classification

    def forward(self, x): return self.fc(x)

def train_local(model, loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for images, labels in loader:
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(model(images), labels)
        loss.backward()
        optimizer.step()

# --- 2. THE HOSPITAL CLIENT (Node) ---
class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id, data_loader):
        self.id = hospital_id
        self.loader = data_loader
        self.model = MedicalModel().to("cuda")

    def fit(self, parameters, config):
        # A. Get Global Weights from Company
        for p, git_p in zip(self.model.parameters(), parameters):
            p.data = torch.from_numpy(git_p)

        # B. Train on Private Patient Data
        train_local(self.model, self.loader)

        # C. BLOCKCHAIN: Upload weights to IPFS & send hash to Sepolia
        # In reality, you'd call a function here:
        # ipfs_hash = upload_to_ipfs(self.model.state_dict())
        # tx_hash = send_to_sepolia(self.id, ipfs_hash)
        print(f"Hospital {self.id} notarized update on Sepolia.")

        return [val.cpu().detach().numpy() for val in self.model.parameters()], len(self.loader.dataset), {}

# --- 3. THE SIMULATION LAUNCHER ---
def client_fn(cid):
    # Simulate different data for each hospital
    dummy_data = DataLoader(TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,))), batch_size=10)
    return HospitalClient(cid, dummy_data)

if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10, 
        client_resources={"num_gpus": 0.2}, # 5 hospitals parallel per GPU
        config=fl.server.ServerConfig(num_rounds=3),
    )