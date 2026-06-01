"""
Asking benefit calculation for the DISP model.
This module implements the asking benefit calculation that helps the model decide
when to ask questions vs. provide answers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from ..config import Config


class AskingBenefitCalculator:
    def __init__(self):
        """Initialize the asking benefit calculator."""
        self.bert_dim = 768  # BERT base hidden size

        # MLP for computing asking benefit
        self.benefit_mlp = nn.Sequential(
            nn.Linear(
                self.bert_dim * 2, 256
            ),  # Input: concatenated query and document embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def compute_benefit(self, query_embedding, doc_embeddings, retrieval_scores):
        """Compute the benefit of asking a question."""
        # Get device from query_embedding
        device = query_embedding.device

        # Debug prints
        print(f"\nDebug - Asking Benefit Inputs:")
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding device: {query_embedding.device}")
        print(f"Doc embeddings shape: {doc_embeddings.shape}")
        print(f"Doc embeddings device: {doc_embeddings.device}")
        print(f"Retrieval scores shape: {retrieval_scores.shape}")
        print(f"Retrieval scores device: {retrieval_scores.device}")

        # Move all tensors to the same device
        query_embedding = query_embedding.to(device)
        doc_embeddings = doc_embeddings.to(device)
        retrieval_scores = retrieval_scores.to(device)

        # Ensure retrieval_scores is 2D
        if retrieval_scores.dim() == 1:
            retrieval_scores = retrieval_scores.unsqueeze(0)  # Add batch dimension

        # Pad or truncate retrieval_scores to match doc_embeddings
        num_docs = doc_embeddings.size(1)
        if retrieval_scores.size(1) < num_docs:
            # Create padding tensor
            padding = torch.zeros(
                retrieval_scores.size(0),
                num_docs - retrieval_scores.size(1),
                device=device,
            )
            retrieval_scores = torch.cat((retrieval_scores, padding), dim=1)
        elif retrieval_scores.size(1) > num_docs:
            retrieval_scores = retrieval_scores[:, :num_docs]

        print(f"\nDebug - After padding retrieval scores:")
        print(f"Retrieval scores shape: {retrieval_scores.shape}")
        print(f"Retrieval scores device: {retrieval_scores.device}")

        # Compute document weights using softmax
        # For 1D tensor, use dim=0, for 2D tensor, use dim=1
        dim = 1 if retrieval_scores.dim() > 1 else 0
        # Convert retrieval scores to float before softmax
        retrieval_scores = retrieval_scores.float()
        doc_weights = F.softmax(retrieval_scores, dim=dim)  # [batch_size, num_docs]

        print(f"\nDebug - After softmax:")
        print(f"Doc weights shape: {doc_weights.shape}")
        print(f"Doc weights device: {doc_weights.device}")

        # Ensure doc_weights has the same batch size as doc_embeddings
        if doc_weights.size(0) != doc_embeddings.size(0):
            doc_weights = doc_weights.repeat(doc_embeddings.size(0), 1)

        print(f"\nDebug - After batch size adjustment:")
        print(f"Doc weights shape: {doc_weights.shape}")
        print(f"Doc embeddings shape: {doc_embeddings.shape}")

        # Compute weighted document embeddings
        weighted_docs = torch.matmul(doc_weights.unsqueeze(1), doc_embeddings).squeeze(
            1
        )  # [batch_size, bert_dim]

        print(f"\nDebug - After weighted sum:")
        print(f"Weighted docs shape: {weighted_docs.shape}")
        print(f"Weighted docs device: {weighted_docs.device}")

        # Compute cosine similarity between query and weighted documents
        similarity = F.cosine_similarity(
            query_embedding, weighted_docs, dim=1
        )  # [batch_size]

        print(f"\nDebug - After similarity:")
        print(f"Similarity shape: {similarity.shape}")
        print(f"Similarity device: {similarity.device}")

        return similarity

    def compute_uncertainty(self, retrieval_scores: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty in retrieval scores."""
        # Get device from retrieval_scores
        device = retrieval_scores.device

        # Debug prints
        print(f"\nDebug - Uncertainty Input:")
        print(f"Retrieval scores shape: {retrieval_scores.shape}")
        print(f"Retrieval scores device: {retrieval_scores.device}")

        # Move tensor to device
        retrieval_scores = retrieval_scores.to(device)

        # Ensure retrieval_scores is 2D
        if retrieval_scores.dim() == 1:
            retrieval_scores = retrieval_scores.unsqueeze(0)  # Add batch dimension

        print(f"\nDebug - After dimension adjustment:")
        print(f"Retrieval scores shape: {retrieval_scores.shape}")

        # Normalize retrieval scores using softmax
        # For 1D tensor, use dim=0, for 2D tensor, use dim=1
        dim = 1 if retrieval_scores.dim() > 1 else 0
        # Convert retrieval scores to float before softmax
        retrieval_scores = retrieval_scores.float()
        probs = F.softmax(retrieval_scores, dim=dim)

        print(f"\nDebug - After softmax:")
        print(f"Probs shape: {probs.shape}")

        # Compute entropy as a measure of uncertainty
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=dim, keepdim=True)

        print(f"\nDebug - After entropy:")
        print(f"Entropy shape: {entropy.shape}")

        # Normalize entropy to [0, 1]
        max_entropy = torch.log(
            torch.tensor(probs.size(dim), dtype=torch.float, device=device)
        )
        normalized_entropy = entropy / max_entropy

        print(f"\nDebug - After normalization:")
        print(f"Normalized entropy shape: {normalized_entropy.shape}")

        return normalized_entropy

    def compute_information_gain(
        self, current_query: torch.Tensor, potential_questions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute potential information gain from asking questions.

        Args:
            current_query (torch.Tensor): Current query embedding [batch_size, bert_dim]
            potential_questions (List[torch.Tensor]): List of potential question embeddings

        Returns:
            torch.Tensor: Information gain scores [batch_size, num_questions]
        """
        batch_size = current_query.size(0)
        num_questions = len(potential_questions)

        # Stack potential questions
        questions = torch.stack(
            potential_questions
        )  # [num_questions, batch_size, bert_dim]
        questions = questions.transpose(0, 1)  # [batch_size, num_questions, bert_dim]

        # Compute cosine similarity between current query and potential questions
        current_query_norm = F.normalize(current_query, p=2, dim=1)
        questions_norm = F.normalize(questions, p=2, dim=2)

        similarity = torch.bmm(
            current_query_norm.unsqueeze(1),  # [batch_size, 1, bert_dim]
            questions_norm.transpose(1, 2),  # [batch_size, bert_dim, num_questions]
        ).squeeze(
            1
        )  # [batch_size, num_questions]

        # Convert similarity to information gain (higher similarity = lower gain)
        information_gain = 1 - similarity

        return information_gain

    def compute_asking_benefit(
        self,
        query_embedding: torch.Tensor,
        doc_embeddings: torch.Tensor,
        retrieval_scores: torch.Tensor,
        potential_questions: List[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the overall asking benefit score."""
        # Get device from query_embedding
        device = query_embedding.device

        # Debug prints
        print(f"\nDebug - Asking Benefit Inputs:")
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Doc embeddings shape: {doc_embeddings.shape}")
        print(f"Retrieval scores shape: {retrieval_scores.shape}")

        # Move all tensors to the same device
        query_embedding = query_embedding.to(device)
        doc_embeddings = doc_embeddings.to(device)
        retrieval_scores = retrieval_scores.to(device)

        # Ensure retrieval_scores is 2D
        if retrieval_scores.dim() == 1:
            retrieval_scores = retrieval_scores.unsqueeze(0)  # Add batch dimension

        # Pad or truncate retrieval_scores to match doc_embeddings
        num_docs = doc_embeddings.size(1)
        if retrieval_scores.size(1) < num_docs:
            # Create padding tensor
            padding = torch.zeros(
                retrieval_scores.size(0),
                num_docs - retrieval_scores.size(1),
                device=device,
            )
            retrieval_scores = torch.cat((retrieval_scores, padding), dim=1)
        elif retrieval_scores.size(1) > num_docs:
            retrieval_scores = retrieval_scores[:, :num_docs]

        print(f"\nDebug - After padding retrieval scores:")
        print(f"Retrieval scores shape: {retrieval_scores.shape}")

        # Compute base benefit
        base_benefit = self.compute_benefit(
            query_embedding, doc_embeddings, retrieval_scores
        )

        # Compute uncertainty
        uncertainty = self.compute_uncertainty(retrieval_scores)

        # Initialize information gain
        information_gain = torch.zeros_like(base_benefit, device=device)
        if potential_questions is not None:
            potential_questions = [q.to(device) for q in potential_questions]
            information_gain = self.compute_information_gain(
                query_embedding, potential_questions
            )

        # Combine benefits
        combined_benefit = (
            base_benefit * 0.4 + uncertainty.squeeze(-1) * 0.3 + information_gain * 0.3
        )

        return {
            "base_benefit": base_benefit,
            "uncertainty": uncertainty,
            "information_gain": information_gain,
            "combined_benefit": combined_benefit,
        }
