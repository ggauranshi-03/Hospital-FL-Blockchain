from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from web3 import Web3
from ml_logic import train_local
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()
origins = [
    "http://localhost:3000",    # React default
    "http://127.0.0.1:3000",    # React alternative
    "http://localhost",
]
# Enable CORS for React
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Explicitly allow these origins
    allow_credentials=True,
    allow_methods=["*"],        # Allow all methods (POST, GET, OPTIONS, etc.)
    allow_headers=["*"],        # Allow all headers
)

# Blockchain Config
w3 = Web3(Web3.HTTPProvider("https://sepolia.gateway.tenderly.co"))
contract_address = "0xFcC28C01206847Be2997A3df882c3aE7EC6aB36b"
# private_key = "" # Hospital's wallet
# account = w3.eth.account.from_key(private_key)

# Minimal ABI
abi = [
    {
        "inputs": [
            {"internalType": "string", "name": "_ipfsHash", "type": "string"},
            {"internalType": "uint256", "name": "_accuracy", "type": "uint256"}
        ],
        "name": "submitUpdate",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

contract = w3.eth.contract(address=contract_address, abi=abi)
class TrainingRequest(BaseModel):
    num_samples: int
@app.post("/start-training")
# async def start_training(req: TrainingRequest):
#     print("Starting ML Training...")
    
#     # 1. Run ML
#     ipfs_hash, accuracy, logs = train_local(req.num_samples)
#     print(f"Training Done. Accuracy: {accuracy}")

#     # 2. Submit to Blockchain
#     accuracy_int = int(accuracy * 100) # Convert 0.85 -> 85
#     print(f"Submitting to Blockchain with IPFS Hash: {ipfs_hash} and Accuracy: {accuracy_int}")
#     # Build Transaction
#     nonce = w3.eth.get_transaction_count(account.address)
#     tx = contract.functions.submitUpdate(ipfs_hash, accuracy_int).build_transaction({
#         'from': account.address,
#         'nonce': nonce,
#         'gas': 500000,
#         'maxFeePerGas': w3.to_wei('50', 'gwei'),
#         'maxPriorityFeePerGas': w3.to_wei('2', 'gwei'),
#     })
    
#     # Sign & Send
#     signed_tx = w3.eth.account.sign_transaction(tx, private_key)
#     tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    
#     return {"status": "Success", "tx_hash": tx_hash.hex(), "accuracy": accuracy}

@app.post("/start-training")
async def start_training(req: TrainingRequest):
    print(f"Starting ML Training with {req.num_samples} samples...")
    
    # 1. Run ML (Returns Hash + Logs)
    ipfs_hash, accuracy, logs = train_local(req.num_samples)
    
    print(f"Training Done. Accuracy: {accuracy}")

    # Return data to Frontend. Frontend will handle the Blockchain Transaction.
    return {
        "status": "Success", 
        "ipfs_hash": ipfs_hash, 
        "accuracy": accuracy, 
        "logs": logs
    }
