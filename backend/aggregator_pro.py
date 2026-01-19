import torch
import requests
import io
import time
import os
from web3 import Web3
from ml_logic import Net, upload_to_pinata # Import upload logic
import json
import copy

# --- CONFIGURATION ---
W3_URL = "https://sepolia.gateway.tenderly.co"
CONTRACT_ADDRESS = "0xFcC28C01206847Be2997A3df882c3aE7EC6aB36b"
OWNER_PRIVATE_KEY = "" # REQUIRED to advance the round
MAX_GLOBAL_ROUNDS = 5       # Set your target rounds
HOSPITAL_THRESHOLD = 2      # Minimum submissions needed

# Parse ABI
ABI_JSON = """[
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
]""" # Use your existing ABI string
ABI = json.loads(ABI_JSON)

w3 = Web3(Web3.HTTPProvider(W3_URL))
account = w3.eth.account.from_key(OWNER_PRIVATE_KEY)
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

def download_model_from_ipfs(ipfs_hash):
    url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return torch.load(io.BytesIO(response.content), map_location=torch.device('cpu'))
    except: return None
    return None

def federated_averaging(model_updates):
    if not model_updates: return None
    global_dict = copy.deepcopy(model_updates[0])
    num_models = len(model_updates)
    for key in global_dict.keys():
        for i in range(1, num_models):
            global_dict[key] += model_updates[i][key]
        global_dict[key] = torch.div(global_dict[key], num_models)
    return global_dict

def advance_blockchain_round():
    """Triggers the startNextRound function on Sepolia"""
    nonce = w3.eth.get_transaction_count(account.address)
    tx = contract.functions.startNextRound().build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': 200000,
        'maxFeePerGas': w3.to_wei('50', 'gwei'),
        'maxPriorityFeePerGas': w3.to_wei('2', 'gwei'),
    })
    signed_tx = w3.eth.account.sign_transaction(tx, OWNER_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"⛓️ Blockchain Round Advanced! Tx: {tx_hash.hex()}")
    return tx_hash

def run_automation_loop():
    print(f"🚀 FL Automation Started. Target Rounds: {MAX_GLOBAL_ROUNDS}")
    
    while True:
        # 1. Check Current Round
        current_round = contract.functions.currentRound().call()
        if current_round > MAX_GLOBAL_ROUNDS:
            print("🏁 Target Global Rounds reached. Stopping process.")
            break

        print(f"--- Monitoring Round {current_round} ---")

        # 2. Get Updates & Check Threshold
        updates = contract.functions.getUpdates(current_round).call()
        print(f"Current Submissions: {len(updates)} / {HOSPITAL_THRESHOLD}")

        if len(updates) >= HOSPITAL_THRESHOLD:
            print(f"✅ Threshold met! Starting Aggregation...")
            
            state_dicts = []
            for update in updates:
                weights = download_model_from_ipfs(update[1])
                if weights: state_dicts.append(weights)

            if len(state_dicts) >= HOSPITAL_THRESHOLD:
                # 3. FedAvg
                global_weights = federated_averaging(state_dicts)
                
                # 4. Save and Upload Global Model
                save_name = f"global_model_r{current_round + 1}.pth"
                torch.save(global_weights, save_name)
                
                print(f"☁️ Uploading Global Model {save_name} to IPFS...")
                global_ipfs_hash = upload_to_pinata(save_name)
                print(f"🌐 Global Hash: {global_ipfs_hash}")

                # 5. Advance Blockchain
                advance_blockchain_round()
                print(f"🎉 Round {current_round} Complete. Moving to {current_round + 1}\n")
            else:
                print("❌ Download failed for some models. Retrying...")
        
        # Wait 30 seconds before checking blockchain again
        time.sleep(30)

if __name__ == "__main__":
    run_automation_loop()