"""
Strategy diversity measurement using Dynamic Time Warping (DTW).
"""

import numpy as np
from typing import List, Dict, Tuple
import torch


class StrategyDiversity:
    def __init__(self):
        """Initialize the strategy diversity measurement."""
        self.strategy_cache = {}  # Cache for storing strategies

    def compute_dtw_distance(
        self, strategy1: List[float], strategy2: List[float]
    ) -> float:
        """
        Compute the DTW distance between two strategies.

        Args:
            strategy1 (List[float]): First strategy sequence (list of action probabilities)
            strategy2 (List[float]): Second strategy sequence (list of action probabilities)

        Returns:
            float: DTW distance between the two strategies
        """
        n, m = len(strategy1), len(strategy2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        # Compute DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(strategy1[i - 1] - strategy2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],  # insertion
                    dtw_matrix[i, j - 1],  # deletion
                    dtw_matrix[i - 1, j - 1],  # match
                )

        return dtw_matrix[n, m]

    def extract_strategy(self, action_probs: List[Dict[str, float]]) -> List[float]:
        """
        Extract strategy from action probabilities.

        Args:
            action_probs (List[Dict[str, float]]): List of action probability dictionaries

        Returns:
            List[float]: Strategy sequence (list of 'ask' probabilities)
        """
        return [probs["ask"] for probs in action_probs]

    def compute_domain_diversity(
        self, domain_strategies: Dict[str, List[Dict[str, float]]]
    ) -> float:
        """
        Compute strategy diversity across domains.

        Args:
            domain_strategies (Dict[str, List[Dict[str, float]]]):
                Dictionary mapping domain names to their strategies

        Returns:
            float: Average DTW distance between all pairs of domain strategies
        """
        # Extract strategies for each domain
        strategies = {
            domain: self.extract_strategy(strategy)
            for domain, strategy in domain_strategies.items()
        }

        # Compute pairwise DTW distances
        n_domains = len(strategies)
        if n_domains < 2:
            return 0.0

        total_distance = 0.0
        count = 0

        # Compare each pair of domains
        domains = list(strategies.keys())
        for i in range(n_domains):
            for j in range(i + 1, n_domains):
                domain1, domain2 = domains[i], domains[j]
                distance = self.compute_dtw_distance(
                    strategies[domain1], strategies[domain2]
                )
                total_distance += distance
                count += 1

        # Return average distance
        return total_distance / count if count > 0 else 0.0

    def compute_turn_diversity(self, strategy: List[Dict[str, float]]) -> float:
        """
        Compute strategy diversity within a single conversation.

        Args:
            strategy (List[Dict[str, float]]): Strategy for a single conversation

        Returns:
            float: Average DTW distance between consecutive turns
        """
        if len(strategy) < 2:
            return 0.0

        # Extract strategy sequence
        strategy_seq = self.extract_strategy(strategy)

        # Compute DTW distance between consecutive turns
        total_distance = 0.0
        for i in range(len(strategy_seq) - 1):
            distance = self.compute_dtw_distance(
                [strategy_seq[i]], [strategy_seq[i + 1]]
            )
            total_distance += distance

        return total_distance / (len(strategy_seq) - 1)

    def evaluate_model_diversity(self, model, dataset_manager) -> Dict[str, float]:
        """
        Evaluate strategy diversity of a model across all domains.

        Args:
            model: The DISP model to evaluate
            dataset_manager: DatasetManager instance

        Returns:
            Dict[str, float]: Dictionary containing diversity metrics
        """
        domain_strategies = {}

        # Collect strategies for each domain
        for domain_name in dataset_manager.get_domain_stats().keys():
            test_data = dataset_manager.get_domain_data(domain_name, split="test")
            strategies = []

            for batch in test_data:
                with torch.no_grad():
                    action_probs = model.get_action_probs(
                        batch["query_history"],
                        batch["documents"],
                        batch["retrieval_scores"],
                    )
                    strategies.append(action_probs)

            domain_strategies[domain_name] = strategies

        # Compute diversity metrics
        metrics = {
            "domain_diversity": self.compute_domain_diversity(domain_strategies),
            "turn_diversity": np.mean(
                [
                    self.compute_turn_diversity(strategy)
                    for strategy in domain_strategies.values()
                ]
            ),
        }

        return metrics
