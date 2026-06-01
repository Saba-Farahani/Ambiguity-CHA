"""
Evaluation metrics implementation for STYLE.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from ..config import Config
import torch


class Metrics:
    @staticmethod
    def compute_sr_at_k(successes: List[bool], k: int = 5) -> float:
        """
        Compute success rate within k turns.

        Args:
            successes: List of success indicators
            k: Number of turns to consider

        Returns:
            Success rate
        """
        if not successes:
            return 0.0

        # Consider only first k turns
        successes_k = successes[:k]
        return sum(successes_k) / len(successes_k)

    @staticmethod
    def compute_avg_turns(turns: List[int]) -> float:
        """
        Compute average number of turns.

        Args:
            turns: List of number of turns per episode

        Returns:
            Average number of turns
        """
        if not turns:
            return 0.0
        return sum(turns) / len(turns)

    @staticmethod
    def compute_recall_at_k(target_ranks: List[int], k: int = 5) -> float:
        """
        Compute recall at k.

        Args:
            target_ranks: List of target document ranks
            k: Number of documents to consider

        Returns:
            Recall at k
        """
        if not target_ranks:
            return 0.0
        return sum(1 for rank in target_ranks if rank < k) / len(target_ranks)

    @staticmethod
    def compute_strategy_diversity(action_sequences):
        """Compute strategy diversity using DTW."""
        if not action_sequences:
            return 0.0

        # Convert CUDA tensors to CPU and then to numpy
        sequences = []
        for seq in action_sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.cpu().numpy()
            sequences.append(seq)

        if len(sequences) < 2:
            return 0.0

        total_distance = 0
        count = 0

        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                try:
                    distance, _ = fastdtw(sequences[i], sequences[j], dist=euclidean)
                    total_distance += distance
                    count += 1
                except Exception as e:
                    print(
                        f"Warning: DTW computation failed for sequences {i} and {j}: {e}"
                    )
                    continue

        return total_distance / count if count > 0 else 0.0

    @staticmethod
    def compute_clarification_benefit(
        ranks_before: List[List[int]], ranks_after: List[List[int]]
    ) -> float:
        """
        Compute clarification benefit.

        Args:
            ranks_before: List of target ranks before clarification
            ranks_after: List of target ranks after clarification

        Returns:
            Average rank improvement
        """
        if not ranks_before or not ranks_after:
            return 0.0

        # Flatten the lists and compute improvements
        improvements = []
        for before_list, after_list in zip(ranks_before, ranks_after):
            if before_list and after_list:  # Check if lists are not empty
                before = before_list[-1]  # Take the last rank before
                after = after_list[-1]  # Take the last rank after
                improvements.append(before - after)

        if not improvements:
            return 0.0

        return sum(improvements) / len(improvements)

    @staticmethod
    def compute_all_metrics(episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            episode_results: List of episode results containing:
                - success: Whether target was found
                - num_turns: Number of turns
                - target_rank: Rank of target document
                - action_sequence: Sequence of actions
                - ranks_before: Target ranks before clarification
                - ranks_after: Target ranks after clarification

        Returns:
            Dictionary of metric names and values
        """
        # Extract metrics
        successes = [result["success"] for result in episode_results]
        turns = [result["num_turns"] for result in episode_results]
        target_ranks = [result["target_rank"] for result in episode_results]
        action_sequences = [result["action_sequence"] for result in episode_results]
        ranks_before = [result["ranks_before"] for result in episode_results]
        ranks_after = [result["ranks_after"] for result in episode_results]

        # Compute metrics
        metrics = {
            "sr@3": Metrics.compute_sr_at_k(successes, k=3),
            "sr@5": Metrics.compute_sr_at_k(successes, k=5),
            "avg_turns": Metrics.compute_avg_turns(turns),
            "recall@3": Metrics.compute_recall_at_k(target_ranks, k=3),
            "recall@5": Metrics.compute_recall_at_k(target_ranks, k=5),
            "strategy_diversity": Metrics.compute_strategy_diversity(action_sequences),
            "clarification_benefit": Metrics.compute_clarification_benefit(
                ranks_before, ranks_after
            ),
        }

        return metrics

    @staticmethod
    def compute_human_evaluation_metrics(
        evaluations: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Compute human evaluation metrics.

        Args:
            evaluations: List of human evaluations containing:
                - helpfulness: Rating of question helpfulness
                - intent_consistency: Rating of intent consistency

        Returns:
            Dictionary of metric names and values
        """
        if not evaluations:
            return {"helpfulness": 0.0, "intent_consistency": 0.0}

        return {
            "helpfulness": sum(eval["helpfulness"] for eval in evaluations)
            / len(evaluations),
            "intent_consistency": sum(
                eval["intent_consistency"] for eval in evaluations
            )
            / len(evaluations),
        }


def compute_metrics(domain_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute overall metrics across all domains."""
    # Initialize metrics
    overall_metrics = {
        "avg_reward": 0.0,
        "success_rate": 0.0,
        "avg_queries": 0.0,
        "avg_responses": 0.0,
        "strategy_diversity": 0.0,
        "domain_transfer": 0.0,
    }

    # Compute average metrics across domains
    num_domains = len(domain_metrics)
    for domain, metrics in domain_metrics.items():
        overall_metrics["avg_reward"] += metrics["reward"] / num_domains
        overall_metrics["success_rate"] += metrics["success"] / num_domains
        overall_metrics["avg_queries"] += metrics["queries"] / num_domains
        overall_metrics["avg_responses"] += metrics["responses"] / num_domains
        overall_metrics["strategy_diversity"] += (
            metrics["strategy_diversity"] / num_domains
        )

    # Compute domain transfer score
    domain_transfer_scores = []
    for domain1 in domain_metrics:
        for domain2 in domain_metrics:
            if domain1 != domain2:
                # Compute transfer score as ratio of metrics
                transfer_score = (
                    domain_metrics[domain2]["success"]
                    / domain_metrics[domain1]["success"]
                )
                domain_transfer_scores.append(transfer_score)

    overall_metrics["domain_transfer"] = np.mean(domain_transfer_scores)

    return overall_metrics


def compute_strategy_diversity(action_sequences: List[List[int]]) -> float:
    """Compute strategy diversity using Dynamic Time Warping."""
    if not action_sequences:
        return 0.0

    # Compute pairwise DTW distances
    distances = []
    for i in range(len(action_sequences)):
        for j in range(i + 1, len(action_sequences)):
            dist = dtw_distance(action_sequences[i], action_sequences[j])
            distances.append(dist)

    # Return average distance
    return np.mean(distances) if distances else 0.0


def dtw_distance(seq1: List[int], seq2: List[int]) -> float:
    """Compute Dynamic Time Warping distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dtw = np.zeros((n + 1, m + 1))
    dtw.fill(np.inf)
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(
                dtw[i - 1, j],  # insertion
                dtw[i, j - 1],  # deletion
                dtw[i - 1, j - 1],  # match
            )

    return dtw[n, m]


def compute_response_quality(target_doc: str, question: str, response: str) -> float:
    """Compute response quality score."""
    # Initialize score
    score = 0.0

    # Check relevance
    if is_relevant(response, target_doc):
        score += 0.4

    # Check clarity
    if is_clear(response):
        score += 0.3

    # Check completeness
    if is_complete(response, question):
        score += 0.3

    return score


def is_relevant(response: str, target_doc: str) -> bool:
    """Check if response is relevant to target document."""
    # TODO: Implement relevance check using BERT similarity
    return True


def is_clear(response: str) -> bool:
    """Check if response is clear and well-structured."""
    # TODO: Implement clarity check using readability metrics
    return True


def is_complete(response: str, question: str) -> bool:
    """Check if response fully answers the question."""
    # TODO: Implement completeness check using question-answering metrics
    return True
