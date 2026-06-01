import random
from collections import namedtuple
import torch
from typing import List, Tuple

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class ReplayMemory:
    def __init__(self, capacity: int):
        """
        Initialize replay memory.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.memory: List[Experience] = []
        self.position = 0

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """
        Store a new experience in memory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences from memory.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.memory) < batch_size:
            raise ValueError(
                f"Not enough experiences in memory. Need {batch_size}, have {len(self.memory)}"
            )

        experiences = random.sample(self.memory, batch_size)

        # Stack tensors and ensure consistent batch dimension
        states = torch.stack([e.state for e in experiences])
        actions = torch.stack([e.action for e in experiences])
        rewards = torch.stack([e.reward for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.stack([e.done for e in experiences])

        # Verify batch dimensions
        batch_dim = states.size(0)
        assert (
            actions.size(0) == batch_dim
        ), f"Actions batch size {actions.size(0)} != states batch size {batch_dim}"
        assert (
            rewards.size(0) == batch_dim
        ), f"Rewards batch size {rewards.size(0)} != states batch size {batch_dim}"
        assert (
            next_states.size(0) == batch_dim
        ), f"Next states batch size {next_states.size(0)} != states batch size {batch_dim}"
        assert (
            dones.size(0) == batch_dim
        ), f"Dones batch size {dones.size(0)} != states batch size {batch_dim}"

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """Return the current size of memory."""
        return len(self.memory)
