"use client";
import { useState, useEffect } from "react";
import { ethers } from "ethers";
import axios from "axios";

const CONTRACT_ADDRESS = "0xFcC28C01206847Be2997A3df882c3aE7EC6aB36b";
const CONTRACT_ABI = [
  "function getUpdates(uint256 _roundId) view returns (tuple(address hospital, string ipfsHash, uint256 accuracy, uint256 roundId)[])",
  "function currentRound() view returns (uint256)"
];

export default function Home() {
  const [updates, setUpdates] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("Idle");

  // 1. Trigger Python Backend
  const startLocalTraining = async () => {
    setLoading(true);
    setStatus("Training ML Model locally...");
    try {
      const res = await axios.post("http://127.0.0.1:8000/start-training");
      setStatus(`Training Complete! Tx: ${res.data.tx_hash.slice(0, 10)}...`);
      fetchUpdates(); // Refresh list
    } catch (err) {
      console.error(err);
      setStatus("Error during training");
    }
    setLoading(false);
  };

  // 2. Fetch Data from Blockchain
  const fetchUpdates = async () => {
    if (typeof window.ethereum !== "undefined") {
      const provider = new ethers.BrowserProvider(window.ethereum);
      const contract = new ethers.Contract(CONTRACT_ADDRESS, CONTRACT_ABI, provider);
      
      try {
        const round = await contract.currentRound();
        const data = await contract.getUpdates(round);
        console.log("Current Round:", round);
        console.log("Fetched Updates:", data);
        setUpdates(data);
      } catch (err) {
        console.error("Blockchain Fetch Error:", err);
      }
    }
  };

  useEffect(() => {
    fetchUpdates();
  }, []);

  return (
    <div className="min-h-screen bg-gray-100 p-10">
      <h1 className="text-3xl font-bold mb-8 text-center text-blue-600">
        🏥 Hospital FL Dashboard
      </h1>

      <div className="grid grid-cols-2 gap-10">
        
        {/* LEFT: Hospital Controls */}
        <div className="bg-white p-6 rounded shadow">
          <h2 className="text-xl font-bold mb-4">Local Node Control</h2>
          <p className="mb-4">Dataset: PneumoniaMNIST (500 local samples)</p>
          <button
            onClick={startLocalTraining}
            disabled={loading}
            className="w-full bg-blue-600 text-white py-3 rounded hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? "Processing..." : "Start Training Round"}
          </button>
          <p className="mt-4 text-sm text-gray-600">Status: {status}</p>
        </div>

        {/* RIGHT: Blockchain Live Feed */}
        <div className="bg-white p-6 rounded shadow">
          <h2 className="text-xl font-bold mb-4">Blockchain Ledger (Sepolia)</h2>
          <div className="overflow-auto max-h-64">
            {updates.length === 0 ? (
              <p className="text-gray-500">No updates for this round yet.</p>
            ) : (
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b">
                    <th className="py-2">Hospital</th>
                    <th>Accuracy</th>
                    <th>Hash</th>
                  </tr>
                </thead>
                <tbody>
                  {updates.map((u, i) => (
                    <tr key={i} className="border-b">
                      <td className="py-2 text-xs font-mono">{u[0].slice(0, 6)}...</td>
                      <td className="text-green-600">{(Number(u[2]) / 100).toFixed(2)}%</td>
                      <td className="text-xs text-gray-500">{u[1]}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          <button onClick={fetchUpdates} className="mt-4 text-blue-500 text-sm underline">
            Refresh Ledger
          </button>
        </div>

      </div>
    </div>
  );
}