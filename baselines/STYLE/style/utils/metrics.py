"""
Evaluation metrics for STYLE implementation.
"""

import numpy as np
from typing import List, Dict, Any
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


def compute_sr_at_k(predictions: List[bool], k: int) -> float:
    """Compute Success Rate at k-th turn."""
    if len(predictions) < k:
        return 0.0
    return float(predictions[k - 1])


def compute_recall_at_k(
    retrieved_docs: List[str], relevant_docs: List[str], k: int
) -> float:
    """Compute Recall@k for retrieved documents."""
    if not retrieved_docs or not relevant_docs:
        return 0.0

    retrieved_at_k = set(retrieved_docs[:k])
    relevant = set(relevant_docs)

    if not relevant:
        return 0.0

    return len(retrieved_at_k.intersection(relevant)) / len(relevant)


def compute_avg_turns(predictions: List[bool]) -> float:
    """Compute average number of turns until success."""
    if not predictions:
        return 0.0

    for i, pred in enumerate(predictions):
        if pred:
            return i + 1

    return len(predictions)


def compute_strategy_diversity(strategies: List[List[bool]]) -> float:
    """Compute strategy diversity using DTW similarity."""
    if len(strategies) < 2:
        return 0.0

    similarities = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            # Convert boolean lists to numeric arrays
            s1 = np.array([1 if x else 0 for x in strategies[i]])
            s2 = np.array([1 if x else 0 for x in strategies[j]])

            # Compute DTW distance
            distance, _ = fastdtw(s1, s2, dist=euclidean)
            similarities.append(distance)

    return np.mean(similarities) if similarities else 0.0


def evaluate_conversation(
    predictions: List[bool],
    retrieved_docs: List[List[str]],
    relevant_docs: List[str],
    k: int = 5,
) -> Dict[str, float]:
    """Evaluate a single conversation."""
    return {
        "sr@k": compute_sr_at_k(predictions, k),
        "recall@5": compute_recall_at_k(
            [doc for docs in retrieved_docs for doc in docs], relevant_docs, 5
        ),
        "avg_turns": compute_avg_turns(predictions),
    }


def evaluate_domain(
    conversations: List[Dict[str, Any]], k: int = 5
) -> Dict[str, float]:
    """Evaluate performance on a domain."""
    metrics = {"sr@k": [], "recall@5": [], "avg_turns": [], "strategy_diversity": []}

    strategies = []

    for conv in conversations:
        # Get conversation metrics
        conv_metrics = evaluate_conversation(
            conv["predictions"], conv["retrieved_docs"], conv["relevant_docs"], k
        )

        # Update metrics
        for k, v in conv_metrics.items():
            metrics[k].append(v)

        # Store strategy for diversity computation
        strategies.append(conv["predictions"])

    # Compute strategy diversity
    metrics["strategy_diversity"] = compute_strategy_diversity(strategies)

    # Average metrics
    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}
