"""
Training stability monitoring utilities for STYLE.
Provides real-time analysis of training metrics and stability indicators.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from collections import deque
import time


class TrainingStabilityMonitor:
    """Monitor training stability and provide real-time analysis."""

    def __init__(self, window_size: int = 100):
        """Initialize the monitor.

        Args:
            window_size: Size of sliding window for trend analysis
        """
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.reward_history = deque(maxlen=window_size)
        self.diversity_history = deque(maxlen=window_size)
        self.gradient_norm_history = deque(maxlen=window_size)
        self.lr_history = deque(maxlen=window_size)
        self.action_distribution_history = deque(maxlen=window_size)

        # Stability thresholds
        self.loss_threshold = 3.0
        self.diversity_threshold = 0.7
        self.gradient_threshold = 1.0
        self.action_skew_threshold = 0.8

    def update(self, metrics: Dict[str, float]):
        """Update monitor with new metrics.

        Args:
            metrics: Dictionary containing training metrics
        """
        # Update histories
        if "loss" in metrics:
            self.loss_history.append(metrics["loss"])
        if "reward" in metrics:
            self.reward_history.append(metrics["reward"])
        if "strategy_diversity" in metrics:
            self.diversity_history.append(metrics["strategy_diversity"])
        if "gradient_norm" in metrics:
            self.gradient_norm_history.append(metrics["gradient_norm"])
        if "learning_rate" in metrics:
            self.lr_history.append(metrics["learning_rate"])

        # Update action distribution
        action_probs = []
        for i in range(3):
            key = f"action_{i}_prob"
            if key in metrics:
                action_probs.append(metrics[key])
        if action_probs:
            self.action_distribution_history.append(action_probs)

    def analyze_stability(self) -> Dict[str, Any]:
        """Analyze current training stability.

        Returns:
            Dictionary containing stability analysis
        """
        analysis = {
            "status": "stable",
            "warnings": [],
            "recommendations": [],
            "metrics": {},
        }

        # Analyze loss stability
        if len(self.loss_history) >= 10:
            recent_losses = list(self.loss_history)[-10:]
            avg_loss = np.mean(recent_losses)
            loss_std = np.std(recent_losses)
            loss_trend = self._compute_trend(recent_losses)

            analysis["metrics"]["loss"] = {
                "current": recent_losses[-1] if recent_losses else 0,
                "average": avg_loss,
                "std": loss_std,
                "trend": loss_trend,
            }

            if avg_loss > self.loss_threshold:
                analysis["warnings"].append(f"High average loss: {avg_loss:.4f}")
                analysis["recommendations"].append("Consider reducing learning rate")
                analysis["status"] = "unstable"

            if loss_std > 1.0:
                analysis["warnings"].append(f"High loss variance: {loss_std:.4f}")
                analysis["recommendations"].append("Consider gradient clipping")

        # Analyze strategy diversity
        if len(self.diversity_history) >= 10:
            recent_diversity = list(self.diversity_history)[-10:]
            avg_diversity = np.mean(recent_diversity)
            diversity_trend = self._compute_trend(recent_diversity)

            analysis["metrics"]["diversity"] = {
                "current": recent_diversity[-1] if recent_diversity else 0,
                "average": avg_diversity,
                "trend": diversity_trend,
            }

            if avg_diversity < self.diversity_threshold:
                analysis["warnings"].append(
                    f"Low strategy diversity: {avg_diversity:.4f}"
                )
                analysis["recommendations"].append("Consider increasing epsilon")
                analysis["status"] = "unstable"

        # Analyze action distribution
        if len(self.action_distribution_history) >= 10:
            recent_distributions = list(self.action_distribution_history)[-10:]
            avg_distribution = np.mean(recent_distributions, axis=0)

            analysis["metrics"]["action_distribution"] = {
                "ask": avg_distribution[0] if len(avg_distribution) > 0 else 0,
                "clarify": avg_distribution[1] if len(avg_distribution) > 1 else 0,
                "answer": avg_distribution[2] if len(avg_distribution) > 2 else 0,
            }

            # Check for action skew
            max_action_prob = max(avg_distribution)
            if max_action_prob > self.action_skew_threshold:
                analysis["warnings"].append(
                    f"Action distribution skewed: max prob = {max_action_prob:.3f}"
                )
                analysis["recommendations"].append(
                    "Consider adjusting reward structure"
                )

        # Analyze gradient norms
        if len(self.gradient_norm_history) >= 10:
            recent_gradients = list(self.gradient_norm_history)[-10:]
            avg_gradient = np.mean(recent_gradients)

            analysis["metrics"]["gradient_norm"] = {
                "current": recent_gradients[-1] if recent_gradients else 0,
                "average": avg_gradient,
            }

            if avg_gradient > self.gradient_threshold:
                analysis["warnings"].append(f"High gradient norm: {avg_gradient:.4f}")
                analysis["recommendations"].append(
                    "Consider stronger gradient clipping"
                )

        return analysis

    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend of a sequence of values.

        Args:
            values: List of numeric values

        Returns:
            Trend description ('increasing', 'decreasing', 'stable')
        """
        if len(values) < 2:
            return "stable"

        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def generate_report(self) -> str:
        """Generate a human-readable stability report.

        Returns:
            Formatted report string
        """
        analysis = self.analyze_stability()

        report = f"""
📊 Training Stability Report
{'='*50}

Status: {'🟢 Stable' if analysis['status'] == 'stable' else '🔴 Unstable'}

📈 Current Metrics:
"""

        # Add metrics
        for metric_name, metric_data in analysis["metrics"].items():
            report += f"  {metric_name.title()}:\n"
            for key, value in metric_data.items():
                if isinstance(value, float):
                    report += f"    {key}: {value:.4f}\n"
                else:
                    report += f"    {key}: {value}\n"

        # Add warnings
        if analysis["warnings"]:
            report += f"\n⚠️ Warnings:\n"
            for warning in analysis["warnings"]:
                report += f"  • {warning}\n"

        # Add recommendations
        if analysis["recommendations"]:
            report += f"\n💡 Recommendations:\n"
            for rec in analysis["recommendations"]:
                report += f"  • {rec}\n"

        return report

    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics for visualization.

        Args:
            save_path: Optional path to save the plot
        """
        if not self.loss_history:
            print("No data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Training Stability Metrics", fontsize=16)

        # Loss plot
        if self.loss_history:
            axes[0, 0].plot(list(self.loss_history))
            axes[0, 0].set_title("Loss")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].axhline(
                y=self.loss_threshold, color="r", linestyle="--", alpha=0.7
            )

        # Reward plot
        if self.reward_history:
            axes[0, 1].plot(list(self.reward_history))
            axes[0, 1].set_title("Reward")
            axes[0, 1].set_ylabel("Reward")

        # Diversity plot
        if self.diversity_history:
            axes[1, 0].plot(list(self.diversity_history))
            axes[1, 0].set_title("Strategy Diversity")
            axes[1, 0].set_ylabel("Diversity")
            axes[1, 0].axhline(
                y=self.diversity_threshold, color="r", linestyle="--", alpha=0.7
            )

        # Action distribution plot
        if self.action_distribution_history:
            distributions = list(self.action_distribution_history)
            if distributions:
                actions = ["Ask", "Clarify", "Answer"]
                avg_dist = np.mean(distributions, axis=0)
                axes[1, 1].bar(actions, avg_dist)
                axes[1, 1].set_title("Action Distribution")
                axes[1, 1].set_ylabel("Probability")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
