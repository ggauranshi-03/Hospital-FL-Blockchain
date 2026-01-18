import torch
from torch.utils.data import DataLoader, TensorDataset
import medmnist
from medmnist import INFO
from ml_logic import Net # Import your architecture

def test_global_model(model_path):
    # 1. Load the Test Dataset (Actual medical testing data)
    info = INFO['pneumoniamnist']
    DataClass = getattr(medmnist, info['python_class'])
    
    # We use split='test' to ensure the model isn't cheating
    test_dataset = DataClass(split='test', download=True, transform=None)
    
    # Process into Tensors
    X = torch.tensor(test_dataset.imgs).float().unsqueeze(1) / 255.0
    y = torch.tensor(test_dataset.labels).long().squeeze()
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # 2. Load the Aggregated Model
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode

    # 3. Perform Testing
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"📊 Global Model Accuracy on Test Set: {accuracy:.2f}%")

if __name__ == "__main__":
    test_global_model("global_model_r3.pth")