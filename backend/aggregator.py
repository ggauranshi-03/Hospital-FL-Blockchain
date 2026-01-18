import torch
import requests
import io
from web3 import Web3
from ml_logic import Net # Import the same architecture
import json
import copy

# --- CONFIG ---
W3_URL = "https://sepolia.gateway.tenderly.co"
CONTRACT_ADDRESS = "0xFcC28C01206847Be2997A3df882c3aE7EC6aB36b"
ABI_JSON = """
[
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": false,
        "inputs": [
            {
                "indexed": true,
                "internalType": "address",
                "name": "hospital",
                "type": "address"
            },
            {
                "indexed": false,
                "internalType": "uint256",
                "name": "roundId",
                "type": "uint256"
            },
            {
                "indexed": false,
                "internalType": "uint256",
                "name": "accuracy",
                "type": "uint256"
            }
        ],
        "name": "UpdateSubmitted",
        "type": "event"
    },
    {
        "inputs": [],
        "name": "currentRound",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "_roundId",
                "type": "uint256"
            }
        ],
        "name": "getUpdates",
        "outputs": [
            {
                "components": [
                    {
                        "internalType": "address",
                        "name": "hospital",
                        "type": "address"
                    },
                    {
                        "internalType": "string",
                        "name": "ipfsHash",
                        "type": "string"
                    },
                    {
                        "internalType": "uint256",
                        "name": "accuracy",
                        "type": "uint256"
                    },
                    {
                        "internalType": "uint256",
                        "name": "roundId",
                        "type": "uint256"
                    }
                ],
                "internalType": "struct FLRegistry.ModelUpdate[]",
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "startNextRound",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
"""

# Convert the string to a Python-readable format
ABI = json.loads(ABI_JSON)

w3 = Web3(Web3.HTTPProvider(W3_URL))
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

def download_model_from_ipfs(ipfs_hash):
    """Downloads the .pth file from Pinata/IPFS"""
    url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
    response = requests.get(url)
    if response.status_code == 200:
        return torch.load(io.BytesIO(response.content), map_location=torch.device('cpu'))
    return None

def federated_averaging(model_updates):
    """
    Implements FedAvg: Mathematical average of all model weights.
    """
    if not model_updates:
        return None

    # Use a deep copy of the first model as the starting point for the average
    # This prevents modifying the original model in the list
    global_dict = copy.deepcopy(model_updates[0])
    
    # Get the number of models to average
    num_models = len(model_updates)
    print(f"Averaging {num_models} valid model updates...")

    for key in global_dict.keys():
        # Sum the weights for this specific layer from all other models
        for i in range(1, num_models):
            global_dict[key] += model_updates[i][key]
        
        # Divide by the number of models to get the mean
        global_dict[key] = torch.div(global_dict[key], num_models)
        
    return global_dict

def run_global_aggregation():
    # 1. Check Current Round
    current_round = contract.functions.currentRound().call()
    print(f"Aggregating Round: {current_round}")

    # 2. Get all updates submitted for this round
    updates = contract.functions.getUpdates(current_round).call()
    
    if len(updates) < 2:
        print("Waiting for more hospitals to submit...")
        return

    # 3. Download all models
    state_dicts = []
    for update in updates:
        hospital_address = update[0]
        ipfs_hash = update[1]
        print(f"Downloading update from {hospital_address}...")
        
        weights = download_model_from_ipfs(ipfs_hash)
        if weights:
            state_dicts.append(weights)

    if len(state_dicts) == 0:
        print("❌ Error: No models were successfully downloaded from IPFS. Check your hashes.")
        return

    # 4. Perform FedAvg
    global_weights = federated_averaging(state_dicts)
    
    if global_weights:
        # 5. Save the new Global Model
        save_name = f"global_model_r{current_round + 1}.pth"
        torch.save(global_weights, save_name)
        print(f"✅ Global Model for Round {current_round + 1} created as '{save_name}'!")
if __name__ == "__main__":
    # You can run this manually or put it in a loop to check every X minutes
    run_global_aggregation()