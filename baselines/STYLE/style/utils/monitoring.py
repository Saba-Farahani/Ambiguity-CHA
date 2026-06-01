"""
Monitoring and logging utilities for STYLE implementation.
"""

import os
import json
import logging
import wandb
from typing import Dict, Any, Optional
from datetime import datetime
from ..config import Config


class Monitor:
    def __init__(
        self,
        project_name: str = "STYLE",
        run_name: Optional[str] = None,
        use_wandb: bool = True,
    ):
        """
        Initialize the monitoring system.

        Args:
            project_name: Name of the project for logging
            run_name: Name of the current run (if None, will use timestamp)
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.project_name = project_name
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.use_wandb = use_wandb

        # Set up logging
        self.logger = logging.getLogger("STYLE")
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        os.makedirs(Config.LOG_DIR, exist_ok=True)

        # Add file handler
        log_file = os.path.join(Config.LOG_DIR, f"{self.run_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config={
                    k: v
                    for k, v in Config.__dict__.items()
                    if not k.startswith("__") and not callable(getattr(Config, k))
                },
            )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to both file and wandb.

        Args:
            metrics: Dictionary of metric names and values
            step: Step number for wandb logging
        """
        # Log to file
        self.logger.info(f"Metrics at step {step}: {json.dumps(metrics, indent=2)}")

        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_retrieval_stats(
        self,
        query: str,
        retrieved_docs: list,
        retrieval_scores: list,
        ambiguity_score: float,
    ):
        """
        Log retrieval statistics.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            retrieval_scores: Retrieval scores
            ambiguity_score: Computed ambiguity score
        """
        stats = {
            "query": query,
            "num_retrieved": len(retrieved_docs),
            "top_score": max(retrieval_scores) if retrieval_scores else 0,
            "ambiguity_score": ambiguity_score,
        }

        self.logger.info(f"Retrieval stats: {json.dumps(stats, indent=2)}")

        if self.use_wandb:
            wandb.log(
                {
                    "retrieval/top_score": stats["top_score"],
                    "retrieval/ambiguity_score": stats["ambiguity_score"],
                }
            )

    def log_clarification_quality(
        self, query: str, clarification_question: str, quality_metrics: Dict[str, float]
    ):
        """
        Log clarification question quality metrics.

        Args:
            query: Original user query
            clarification_question: Generated clarification question
            quality_metrics: Dictionary of quality metrics
        """
        self.logger.info(f"Clarification quality for query '{query}':")
        self.logger.info(f"Question: {clarification_question}")
        self.logger.info(f"Metrics: {json.dumps(quality_metrics, indent=2)}")

        if self.use_wandb:
            wandb.log(
                {
                    f"clarification/{metric}": value
                    for metric, value in quality_metrics.items()
                }
            )

    def log_training_progress(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: Dict[str, float],
    ):
        """
        Log training progress.

        Args:
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_metrics: Validation metrics
        """
        self.logger.info(f"Epoch {epoch}:")
        self.logger.info(f"Train loss: {train_loss:.4f}")
        self.logger.info(f"Val loss: {val_loss:.4f}")
        self.logger.info(f"Val metrics: {json.dumps(val_metrics, indent=2)}")

        if self.use_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    **{f"val/{metric}": value for metric, value in val_metrics.items()},
                },
                step=epoch,
            )

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log an error with optional context.

        Args:
            error: Exception that occurred
            context: Optional context information
        """
        error_msg = f"Error: {str(error)}"
        if context:
            error_msg += f"\nContext: {json.dumps(context, indent=2)}"

        self.logger.error(error_msg)

        if self.use_wandb:
            wandb.log({"error/message": str(error), "error/context": context or {}})

    def finish(self):
        """Finish logging and cleanup."""
        if self.use_wandb:
            wandb.finish()
