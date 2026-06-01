"""
Build domain-invariant state vectors: s_t = [H_t || D_t || score^{1:k}_t].
"""

from typing import List, Union
import torch
from ..config import Config


class StateBuilder:
    """Construct DISP input from conversation history, documents, and scores."""

    def __init__(self, encoder, config: Config):
        self.encoder = encoder
        self.k = config.TOP_K_DOCS
        self.device = config.DEVICE

    def build(
        self,
        query_history: List[str],
        documents: List[str],
        retrieval_scores: List[float],
    ) -> torch.Tensor:
        """
        Args:
            query_history: Utterances in the dialogue so far.
            documents: Top-k retrieved document strings (or IDs resolved upstream).
            retrieval_scores: Top-k retrieval scores.

        Returns:
            State tensor of shape [1, state_input_dim].
        """
        history_text = " [SEP] ".join(query_history) if query_history else ""
        h_enc = self.encoder.encode([history_text])  # (1, hidden)

        docs = list(documents[: self.k]) if documents else []
        while len(docs) < self.k:
            docs.append("")

        d_enc = self.encoder.encode(docs)  # (k, hidden)
        d_flat = d_enc.view(1, -1)

        scores = list(retrieval_scores[: self.k]) if retrieval_scores else []
        while len(scores) < self.k:
            scores.append(0.0)

        score_tensor = torch.tensor(
            scores, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        state = torch.cat([h_enc.to(self.device), d_flat.to(self.device), score_tensor], dim=1)
        return state

    def zero_state(self) -> torch.Tensor:
        dim = Config.state_input_dim(self.encoder.hidden_size, self.k)
        return torch.zeros((1, dim), device=self.device)
