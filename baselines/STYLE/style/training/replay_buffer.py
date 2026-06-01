"""
Replay Buffer implementation for DQN training.
"""

import random
import torch
from collections import deque
from typing import Dict, List, Tuple, Any


class ReplayBuffer:
    def __init__(self, capacity=10000):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum size of the buffer (default: 10000 as per paper)
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(
        self,
        state: Dict[str, Any],
        action: Any,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ):
        """Store a transition in the buffer."""
        # Convert everything to tensors before storing
        if isinstance(reward, (int, float)):
            reward = torch.tensor([reward], dtype=torch.float)
        if isinstance(done, bool):
            done = torch.tensor([done], dtype=torch.float)
        if isinstance(action, (int, float)):
            action = torch.tensor([action], dtype=torch.float)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)

        # Properly format the batch
        return {
            "state": {
                "query_history": [s["query_history"] for s in states],
                "documents": [s["documents"] for s in states],
                "retrieval_scores": torch.stack(
                    [s["retrieval_scores"] for s in states]
                ),
            },
            "action": torch.stack(actions),
            "reward": torch.stack(rewards),
            "next_state": {
                "query_history": [s["query_history"] for s in next_states],
                "documents": [s["documents"] for s in next_states],
                "retrieval_scores": torch.stack(
                    [s["retrieval_scores"] for s in next_states]
                ),
            },
            "done": torch.stack(dones),
        }

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
