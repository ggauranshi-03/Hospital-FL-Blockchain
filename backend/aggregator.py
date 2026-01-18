import torch
import requests
import io
from web3 import Web3
from ml_logic import Net # Import the same architecture
import json

# --- CONFIG ---
W3_URL = "https://sepolia.gateway.tenderly.co"
CONTRACT_ADDRESS = "0xFcC28C01206847Be2997A3df882c3aE7EC6aB36b"
ABI = [
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
		"name": "owner",
		"outputs": [
			{
				"internalType": "address",
				"name": "",
				"type": "address"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "roundUpdates",
		"outputs": [
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
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "startNextRound",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "_ipfsHash",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "_accuracy",
				"type": "uint256"
			}
		],
		"name": "submitUpdate",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	}
]

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
    model_updates: List of state_dicts
    """
    global_dict = model_updates[0]
    for key in global_dict.keys():
        for i in range(1, len(model_updates)):
            global_dict[key] += model_updates[i][key]
        global_dict[key] = torch.div(global_dict[key], len(model_updates))
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

    # 4. Perform FedAvg
    global_weights = federated_averaging(state_dicts)
    
    # 5. Save the new Global Model
    torch.save(global_weights, f"global_model_r{current_round + 1}.pth")
    print(f"✅ Global Model for Round {current_round + 1} created!")