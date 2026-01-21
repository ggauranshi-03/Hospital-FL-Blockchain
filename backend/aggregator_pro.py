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
    """Downloads the .pth file with Rate-Limit (429) handling."""
    # 1. Skip known invalid hashes immediately
    if "Fake" in ipfs_hash or ipfs_hash == "QmHash748":
        print(f"⏩ Skipping known invalid CID: {ipfs_hash}")
        return None

    url = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code == 200:
                return torch.load(io.BytesIO(response.content), map_location=torch.device('cpu'))
            
            elif response.status_code == 429:
                # Rate limit hit - wait and try again
                print(f"⏳ Rate limit [429] hit for {ipfs_hash}. Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2 # Exponential backoff
                
            else:
                print(f"⚠️ IPFS Download Error [{response.status_code}] for CID: {ipfs_hash}")
                return None
                
        except Exception as e:
            print(f"❌ Network Exception for CID {ipfs_hash}: {e}")
            return None
    
    print(f"🛑 Failed to download {ipfs_hash} after {max_retries} attempts.")
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
        try:
            current_round = contract.functions.currentRound().call()
            if current_round > MAX_GLOBAL_ROUNDS:
                print("🏁 Target Global Rounds reached. Stopping process.")
                break

            print(f"\n--- Monitoring Round {current_round} ---")
            all_updates = contract.functions.getUpdates(current_round).call()
            
            if len(all_updates) >= HOSPITAL_THRESHOLD:
                print(f"✅ Submissions detected ({len(all_updates)}). Filtering and downloading...")
                
                valid_state_dicts = []
                for update in all_updates:
                    hospital_addr = update[0]
                    cid = update[1]
                    
                    weights = download_model_from_ipfs(cid)
                    
                    if weights:
                        valid_state_dicts.append(weights)
                        print(f"  📥 Successfully loaded model from {hospital_addr[:10]}")
                    
                    # Small sleep between different models to avoid triggering rate limits
                    time.sleep(2) 

                # FINAL CHECK: Did we get enough GOOD models?
                if len(valid_state_dicts) >= HOSPITAL_THRESHOLD:
                    print(f"🔥 Threshold met with {len(valid_state_dicts)} VALID models. Aggregating...")
                    
                    global_weights = federated_averaging(valid_state_dicts)
                    save_name = f"global_model_r{current_round + 1}.pth"
                    torch.save(global_weights, save_name)
                    
                    print(f"☁️ Uploading to IPFS...")
                    global_ipfs_hash = upload_to_pinata(save_name)
                    
                    advance_blockchain_round()
                    print(f"🎉 Round {current_round} Complete!")
                else:
                    print(f"🛑 Only {len(valid_state_dicts)} valid models found. Need {HOSPITAL_THRESHOLD}.")
            
            time.sleep(30)
        except Exception as e:
            print(f"🚨 Loop Error: {e}")
            time.sleep(10)
if __name__ == "__main__":
    run_automation_loop()
