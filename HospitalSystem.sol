// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLRegistry {
    struct ModelUpdate {
        address hospital;
        string ipfsHash;
        uint256 accuracy; // Stored as integer (e.g., 8550 = 85.50%)
        uint256 roundId;
    }

    mapping(uint256 => ModelUpdate[]) public roundUpdates;
    uint256 public currentRound = 1;
    address public owner;

    event UpdateSubmitted(address indexed hospital, uint256 roundId, uint256 accuracy);

    constructor() {
        owner = msg.sender;
    }

    function startNextRound() external {
        require(msg.sender == owner, "Only owner can start round");
        currentRound++;
    }

    function submitUpdate(string memory _ipfsHash, uint256 _accuracy) external {
        roundUpdates[currentRound].push(ModelUpdate({
            hospital: msg.sender,
            ipfsHash: _ipfsHash,
            accuracy: _accuracy,
            roundId: currentRound
        }));
        emit UpdateSubmitted(msg.sender, currentRound, _accuracy);
    }

    function getUpdates(uint256 _roundId) external view returns (ModelUpdate[] memory) {
        return roundUpdates[_roundId];
    }
}