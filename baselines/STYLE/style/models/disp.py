"""
DISP (Domain-Invariant Strategy Planner) implementation with Dueling DQN architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .bert_encoder import BERTEncoder
from .state_builder import StateBuilder
from ..config import Config
import random
from collections import namedtuple, deque

# Define the experience tuple
Experience = namedtuple(
    "Experience", ("state", "action", "next_state", "reward", "done")
)


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Add a new experience to the buffer.

        Args:
            state: Current state tensor
            action: Action tensor
            reward: Reward value
            next_state: Next state tensor
            done: Whether episode is done
        """
        # Ensure all tensors are on the correct device
        state = state.to(self.device)
        next_state = next_state.to(self.device)

        # Ensure state tensors have correct shape [1, features]
        if state.dim() == 3:
            state = state.squeeze(0)  # Remove batch dimension if present
        if next_state.dim() == 3:
            next_state = next_state.squeeze(0)  # Remove batch dimension if present

        # Handle action tensor - ensure it's the right shape for gather operation
        if isinstance(action, torch.Tensor):
            action = action.to(self.device)
            # Ensure action has shape [1] for single action or [batch_size] for batch
            if action.dim() == 0:
                action = action.unsqueeze(0)  # [1]
        else:
            # Convert scalar action to tensor
            action = torch.tensor([action], device=self.device, dtype=torch.long)

        # Convert reward and done to tensors
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        done = torch.tensor([done], device=self.device, dtype=torch.bool)

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Dictionary containing batched tensors
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Buffer size ({len(self.buffer)}) is smaller than batch_size ({batch_size})"
            )

        batch = random.sample(self.buffer, batch_size)

        # Stack tensors and ensure they're on the correct device
        states = torch.stack([exp["state"] for exp in batch]).to(self.device)
        next_states = torch.stack([exp["next_state"] for exp in batch]).to(self.device)
        rewards = torch.stack([exp["reward"] for exp in batch]).to(self.device)
        dones = torch.stack([exp["done"] for exp in batch]).to(self.device)

        # Handle actions carefully - ensure they have shape [batch_size, 1]
        actions = []
        for exp in batch:
            action = exp["action"]
            if action.dim() == 0:
                # Scalar action: [] -> [1]
                action = action.unsqueeze(0)
            elif action.dim() == 1 and action.size(0) == 1:
                # Single action: [1] -> [1] (already correct)
                pass
            else:
                # Reshape to [1]
                action = action.view(1)
            actions.append(action)

        actions = torch.stack(actions).to(self.device)

        return {
            "state": states,
            "action": actions,
            "reward": rewards,
            "next_state": next_states,
            "done": dones,
        }

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class DuelingDQN(nn.Module):
    """Dueling DQN architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, state):
        """
        Forward pass.

        Args:
            state: Input tensor [batch_size, input_dim]

        Returns:
            Q-values for each action [batch_size, num_actions]
        """
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.feature_layer(state)
        values = self.value_stream(features)  # [batch_size, 1]
        advantages = self.advantage_stream(features)  # [batch_size, num_actions]

        # Combine value and advantage streams
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class DQN(nn.Module):
    """Deep Q-Network for DISP."""

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        """
        Initialize DQN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_actions: Number of possible actions
        """
        super().__init__()

        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Q-value prediction layer
        self.q_layer = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Q-values for each action [batch_size, num_actions]
        """
        features = self.feature_layer(x)
        q_values = self.q_layer(features)
        return q_values


class DISP(nn.Module):
    """DISP model for information-seeking conversations."""

    def __init__(self, config: Config):
        """Initialize DISP model."""
        super().__init__()
        self.config = config
        self.device = config.DEVICE

        self.bert_encoder = BERTEncoder(config=config, device=self.device)
        self.state_builder = StateBuilder(self.bert_encoder, config)

        input_dim = config.state_input_dim(
            self.bert_encoder.hidden_size, config.TOP_K_DOCS
        )

        self.dqn = DuelingDQN(
            input_dim=input_dim,
            hidden_dim=config.DISP_HIDDEN_SIZE,
            output_dim=config.NUM_ACTIONS,
        ).to(self.device)

        self.target_dqn = DuelingDQN(
            input_dim=input_dim,
            hidden_dim=config.DISP_HIDDEN_SIZE,
            output_dim=config.NUM_ACTIONS,
        ).to(self.device)

        self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=config.LEARNING_RATE)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(config.MEMORY_SIZE)

        # Initialize action history
        self.action_history = []

        # Initialize training step counter
        self.steps_done = 0

        # Set device for all components
        self.to(self.device)

    def select_action(
        self, state: torch.Tensor, eval_mode: bool = False
    ) -> torch.Tensor:
        """Select action using epsilon-greedy policy or greedy policy for evaluation.

        Args:
            state: State tensor of shape [batch_size, state_dim]
            eval_mode: If True, use greedy policy (no exploration)

        Returns:
            Action tensor of shape [batch_size]
        """
        # Ensure state is on the correct device
        state = state.to(self.device)

        # Get action values
        with torch.no_grad():
            action_values = self.dqn(state)

        # Select action using epsilon-greedy policy or greedy policy
        if eval_mode or random.random() >= self.config.EPSILON:
            # Greedy action selection (for evaluation or when epsilon condition is met)
            action = action_values.argmax(dim=1)
        else:
            # Epsilon-greedy exploration (for training)
            action = torch.randint(0, self.config.NUM_ACTIONS, (1,), device=self.device)

        # Store action in history
        self.action_history.append(action.item())

        return action

    def train_on_batch(self) -> Optional[float]:
        """Train on a batch of experiences.

        Returns:
            Loss value if training was successful, None otherwise
        """
        try:
            # Sample batch
            if len(self.replay_buffer) < self.config.BATCH_SIZE:
                return None

            batch = self.replay_buffer.sample(self.config.BATCH_SIZE)

            # All tensors should already be on the correct device from ReplayBuffer
            state = batch["state"]
            action = batch["action"]
            reward = batch["reward"]
            next_state = batch["next_state"]
            done = batch["done"]

            # Debug tensor shapes
            if self.config.DEBUG:
                print(f"[DEBUG] State shape: {state.shape}")
                print(f"[DEBUG] Action shape: {action.shape}")
                print(f"[DEBUG] Action type: {type(action)}")
                print(f"[DEBUG] Action content: {action}")

            # Fix state shape if it has extra dimensions
            if state.dim() == 3:
                state = state.squeeze(1)
                if self.config.DEBUG:
                    print(f"[DEBUG] Squeezed state shape: {state.shape}")

            # Fix next_state shape if it has extra dimensions
            if next_state.dim() == 3:
                next_state = next_state.squeeze(1)
                if self.config.DEBUG:
                    print(f"[DEBUG] Squeezed next_state shape: {next_state.shape}")

            # Ensure action has the right shape for gathering [batch_size, 1]
            if action.dim() == 1:
                if action.size(0) == 1:
                    action = action.unsqueeze(1)
                else:
                    action = action.unsqueeze(1)
            elif action.dim() == 0:
                action = action.unsqueeze(0).unsqueeze(1)
            else:
                action = action.view(-1, 1)

            if self.config.DEBUG:
                print(f"[DEBUG] Final action shape: {action.shape}")

            q_values = self.dqn(state)
            if self.config.DEBUG:
                print(f"[DEBUG] Q values shape: {q_values.shape}")

            current_q_values = q_values.gather(1, action)

            # Get next Q values from target network
            with torch.no_grad():
                next_q_values = self.target_dqn(next_state).max(1)[0]
                target_q_values = (
                    reward.squeeze()
                    + (1 - done.squeeze().float()) * self.config.GAMMA * next_q_values
                )

            # Compute loss
            loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            # Enhanced gradient clipping for stability
            max_grad_norm = 1.0  # Reduced from 100 for better stability
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.dqn.parameters(), max_grad_norm
            )

            # Log gradient norm if debug mode
            if self.config.DEBUG:
                print(f"[DEBUG] Gradient norm: {grad_norm:.4f}")

            # Check for gradient explosion
            if grad_norm > max_grad_norm * 0.9:  # If clipping was applied
                print(f"⚠️ Gradient clipping applied: norm = {grad_norm:.4f}")

            self.optimizer.step()

            # Decay epsilon
            if self.config.EPSILON > self.config.EPSILON_MIN:
                self.config.EPSILON *= self.config.EPSILON_DECAY

            # Update target network periodically
            self.steps_done += 1
            if self.steps_done % self.config.TARGET_UPDATE == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())
                if self.config.DEBUG:
                    print(f"[DEBUG] Target network updated at step {self.steps_done}")

            return loss.item()

        except Exception as e:
            print(f"Error in train_on_batch: {e}")
            return None

    def get_action_history(self) -> List[int]:
        """Get action history."""
        return self.action_history

    def construct_features(
        self,
        query_history: List[str],
        retrieved_docs: List[Any],
        retrieval_scores: List[float],
    ) -> torch.Tensor:
        """Construct paper state vector [H_t || D_t || score^{1:k}]."""
        if isinstance(query_history, str):
            query_history = [query_history]
        doc_texts = []
        for doc in retrieved_docs:
            doc_texts.append(str(doc) if doc is not None else "")
        if not retrieval_scores:
            retrieval_scores = [1.0 / (i + 1) for i in range(len(doc_texts))]
        return self.state_builder.build(query_history, doc_texts, retrieval_scores)

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def decide(self, query_history, documents, retrieval_scores):
        """Return action index: 0=answer, 1=ask."""
        features = self.construct_features(query_history, documents, retrieval_scores)
        with torch.no_grad():
            q_values = self.dqn(features.to(self.device))
        return q_values.argmax(dim=1).item()

    def get_action_probs(self, query_history, documents, retrieval_scores):
        """Return Q-values for [answer, ask]."""
        features = self.construct_features(query_history, documents, retrieval_scores)
        return self.dqn(features.to(self.device))

    def forward(self, *args, **kwargs):
        """Forward pass for the DISP model.
        
        This is the main forward function that the model loading expects.
        Handles different calling patterns flexibly.
        
        Args:
            *args: Variable arguments (query_history, documents, etc.)
            **kwargs: Keyword arguments (retrieval_scores, etc.)
            
        Returns:
            Action probabilities tensor
        """
        # Handle different calling patterns
        if len(args) == 1:
            # Called with single argument (likely state tensor)
            state = args[0]
            if isinstance(state, torch.Tensor):
                # If it's a tensor, use it directly with DQN
                return self.dqn(state)
            else:
                # If it's a string, treat as query
                query_history = [state] if isinstance(state, str) else state
                documents = kwargs.get('documents', [''])
                retrieval_scores = kwargs.get('retrieval_scores', [1.0])
        elif len(args) == 2:
            # Called with two arguments (query_history, documents)
            query_history, documents = args
            retrieval_scores = kwargs.get('retrieval_scores', [1.0])
        else:
            # Called with three arguments (query_history, documents, retrieval_scores)
            query_history = args[0] if len(args) > 0 else kwargs.get('query_history', [''])
            documents = args[1] if len(args) > 1 else kwargs.get('documents', [''])
            retrieval_scores = args[2] if len(args) > 2 else kwargs.get('retrieval_scores', [1.0])
        
        # Ensure we have valid inputs
        if not query_history:
            query_history = ['']
        if not documents:
            documents = ['']
        if not retrieval_scores:
            retrieval_scores = [1.0]
        
        # Use the existing get_action_probs method
        return self.get_action_probs(query_history, documents, retrieval_scores)

    def _prepare_state(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Prepare state tensor from batch."""
        try:
            query_history = batch.get("query_history", [])
            documents = batch.get("documents", [])
            retrieval_scores = batch.get("retrieval_scores", [])

            if isinstance(retrieval_scores, torch.Tensor):
                retrieval_scores = retrieval_scores.detach().cpu().tolist()

            if not query_history:
                query_history = [batch.get("query", "What information are you looking for?")]
            if not documents:
                documents = ["This document contains general information and resources."]

            return self.construct_features(query_history, documents, retrieval_scores)

        except Exception as e:
            if self.config.DEBUG:
                print(f"Error preparing state: {e}")
            return self.state_builder.zero_state()

    def save(self, path: str):
        """Save the model state."""
        torch.save(
            {
                "dqn": self.dqn.state_dict(),
                "target_dqn": self.target_dqn.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.config.EPSILON,
            },
            path,
        )

    def load(self, path: str):
        """Load the model state."""
        checkpoint = torch.load(path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "dqn" in checkpoint:
            # New format with dictionary keys
            self.dqn.load_state_dict(checkpoint["dqn"])
            self.target_dqn.load_state_dict(checkpoint["target_dqn"])
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if "epsilon" in checkpoint:
                self.config.EPSILON = checkpoint["epsilon"]
        else:
            # Old format with direct weights
            # Filter checkpoint keys for DQN and target DQN
            dqn_state_dict = {}
            target_dqn_state_dict = {}

            for key, value in checkpoint.items():
                if key.startswith("dqn."):
                    # Remove 'dqn.' prefix for state_dict
                    dqn_key = key[4:]  # Remove 'dqn.' prefix
                    dqn_state_dict[dqn_key] = value
                elif key.startswith("target_dqn."):
                    # Remove 'target_dqn.' prefix for state_dict
                    target_key = key[12:]  # Remove 'target_dqn.' prefix
                    target_dqn_state_dict[target_key] = value

            # Load the state dictionaries
            if dqn_state_dict:
                self.dqn.load_state_dict(dqn_state_dict)
            if target_dqn_state_dict:
                self.target_dqn.load_state_dict(target_dqn_state_dict)

            # Set a reasonable epsilon value for old models
            self.config.EPSILON = 0.1  # Conservative epsilon for evaluation
