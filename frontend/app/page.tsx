"use client";
import { useState, useEffect } from "react";
import { ethers } from "ethers";
import axios from "axios";

// TypeScript Window Fix
declare global { interface Window { ethereum?: any; } }

const CONTRACT_ADDRESS = "0xFcC28C01206847Be2997A3df882c3aE7EC6aB36b";
const CONTRACT_ABI = [
  "function getUpdates(uint256 _roundId) view returns (tuple(address hospital, string ipfsHash, uint256 accuracy, uint256 roundId)[])",
  "function currentRound() view returns (uint256)",
  "function submitUpdate(string memory _ipfsHash, uint256 _accuracy)"
];

export default function Home() {
  // State
  const [account, setAccount] = useState("");
  const [sampleSize, setSampleSize] = useState(500);
  const [logs, setLogs] = useState<string[]>([]);
  const [updates, setUpdates] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Idle");

  // 1. Connect Wallet (Requirement 1)
  const connectWallet = async () => {
    if (typeof window.ethereum !== "undefined") {
      try {
        const provider = new ethers.BrowserProvider(window.ethereum);
        const accounts = await provider.send("eth_requestAccounts", []);
        setAccount(accounts[0]);
        fetchUpdates(); // Load ledger on connect
      } catch (err) {
        console.error(err);
      }
    } else {
      alert("Please install MetaMask!");
    }
  };

  // 2. Start Training & Submit to Chain
  const startTrainingProcess = async () => {
    if (!account) return alert("Connect Wallet first!");
    
    setLoading(true);
    setLogs([]); // Clear old logs
    setStatus("⏳ Training Model (10 Epochs)...");

    try {
      // A. Call Python Backend (Compute)
      const res = await axios.post("http://127.0.0.1:8000/start-training", {
        num_samples: Number(sampleSize)
      });

      const { ipfs_hash, accuracy, logs } = res.data;
      setLogs(logs); // Show logs in UI (Requirement 3)
      setStatus("✅ Training Done. Uploading to Blockchain...");

      // B. Submit to Smart Contract (Frontend Signing)
      const provider = new ethers.BrowserProvider(window.ethereum);
      const signer = await provider.getSigner();
      const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, signer);

      const accuracyInt = Math.floor(accuracy * 10000); // 0.85 -> 8500
      
      const tx = await contract.submitUpdate(ipfs_hash, accuracyInt);
      setStatus(`waiting for tx: ${tx.hash}...`);
      await tx.wait();
      
      setStatus(`🎉 Success! Hash: ${ipfs_hash}`);
      fetchUpdates(); // Refresh Ledger

    } catch (err) {
      console.error(err);
      setStatus("❌ Error occurred (Check Console)");
    }
    setLoading(false);
  };

  // 3. Fetch Ledger Updates
  const fetchUpdates = async () => {
    if (window.ethereum) {
      const provider = new ethers.BrowserProvider(window.ethereum);
      const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);
      try {
        const round = await contract.currentRound();
        const data = await contract.getUpdates(round);
        setUpdates(data);
      } catch (err) { console.error(err); }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-10 font-sans text-gray-800">
      
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold text-blue-700">🏥 Decentralized FL Node</h1>
        {account ? (
          <span className="bg-green-100 text-green-800 px-4 py-2 rounded-full font-mono">
            {account.slice(0, 6)}...{account.slice(-4)}
          </span>
        ) : (
          <button onClick={connectWallet} className="bg-orange-500 text-white px-6 py-2 rounded shadow hover:bg-orange-600">
            Connect Wallet
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* LEFT PANEL: Controls */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
          <h2 className="text-xl font-bold mb-4 border-b pb-2">1. Local Training</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">Data Samples to Use</label>
            <input 
              type="number" 
              value={sampleSize} 
              onChange={(e) => setSampleSize(Number(e.target.value))}
              className="w-full p-2 border rounded focus:ring-2 focus:ring-blue-500 outline-none"
            />
            <p className="text-xs text-gray-500 mt-1">Randomly selected from PneumoniaMNIST</p>
          </div>

          <button
            onClick={startTrainingProcess}
            disabled={loading || !account}
            className={`w-full py-3 rounded-lg font-bold text-white transition ${
              loading || !account ? "bg-gray-400 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
            }`}
          >
            {loading ? "Training & Uploading..." : "Start Training Round"}
          </button>
          
          <div className="mt-4 p-3 bg-gray-100 rounded text-sm min-h-[40px] break-all">
            <strong>Status:</strong> {status}
          </div>

          {/* LOGS WINDOW (Requirement 3) */}
          <div className="mt-4">
            <h3 className="text-sm font-bold mb-2">Training Logs</h3>
            <div className="bg-black text-green-400 p-3 rounded h-48 overflow-y-auto font-mono text-xs">
              {logs.length === 0 ? <p className="text-gray-500">Logs will appear here...</p> : logs.map((log, i) => (
                <div key={i}>{log}</div>
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT PANEL: Blockchain Ledger */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-200">
          <h2 className="text-xl font-bold mb-4 border-b pb-2">2. Global Ledger (Sepolia)</h2>
          <div className="overflow-auto h-[500px]">
            {updates.length === 0 ? (
              <div className="text-center text-gray-500 mt-10">No updates submitted for this round.</div>
            ) : (
              <table className="w-full text-left">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="p-3 text-sm text-gray-500">Hospital</th>
                    <th className="p-3 text-sm text-gray-500">Acc</th>
                    <th className="p-3 text-sm text-gray-500">Model Hash</th>
                  </tr>
                </thead>
                <tbody>
                  {updates.map((u, i) => (
                    <tr key={i} className="border-b hover:bg-gray-50">
                      <td className="p-3 font-mono text-xs text-blue-600">
                        {u[0].slice(0, 6)}...
                      </td>
                      <td className="p-3 font-bold text-green-600">
                        {(Number(u[2]) / 100).toFixed(2)}%
                      </td>
                      <td className="p-3">
                        <a 
                          href={`https://gateway.pinata.cloud/ipfs/${u[1]}`} 
                          target="_blank" 
                          className="text-xs text-purple-600 underline"
                        >
                          {u[1].slice(0, 8)}...
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          <button onClick={fetchUpdates} className="mt-4 text-sm text-blue-500 hover:underline w-full text-center">
            Refresh Ledger Data
          </button>
        </div>
      </div>
    </div>
  );
}