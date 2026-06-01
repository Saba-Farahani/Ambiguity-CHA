import os
import json
import logging
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class TrainingLogger:
    """Logger for training monitoring and debugging."""

    def __init__(self, config, log_dir: str = "logs"):
        """Initialize logger."""
        self.config = config
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory
        self.run_dir = os.path.join(log_dir, f"run_{self.timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=self.run_dir)

        # Initialize file logger
        self.logger = logging.getLogger("training")
        self.logger.setLevel(logging.INFO)

        # Add file handler
        fh = logging.FileHandler(os.path.join(self.run_dir, "training.log"))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Initialize metrics storage
        self.metrics = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }

        # Save config
        self._save_config()

    def log_epoch(
        self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ):
        """Log metrics for an epoch."""
        # Log to tensorboard
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f"train/{metric_name}", value, epoch)

        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f"val/{metric_name}", value, epoch)

        # Log to file
        self.logger.info(f"\nEpoch {epoch + 1}")
        self.logger.info("Training Metrics:")
        for metric_name, value in train_metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")

        self.logger.info("\nValidation Metrics:")
        for metric_name, value in val_metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")

        # Store metrics
        for metric_name, value in train_metrics.items():
            self.metrics["train"][metric_name].append(value)

        for metric_name, value in val_metrics.items():
            self.metrics["val"][metric_name].append(value)

    def log_batch(self, epoch: int, batch_idx: int, batch_metrics: Dict[str, float]):
        """Log metrics for a batch."""
        # Log to tensorboard
        for metric_name, value in batch_metrics.items():
            self.writer.add_scalar(
                f"train/batch/{metric_name}",
                value,
                epoch * self.config.BATCHES_PER_EPOCH + batch_idx,
            )

    def log_action(
        self, action: int, reward: float, next_state: torch.Tensor, done: bool
    ):
        """Log action details for debugging."""
        self.logger.debug(f"Action: {action}")
        self.logger.debug(f"Reward: {reward:.4f}")
        self.logger.debug(f"Done: {done}")
        self.logger.debug(f"Next state shape: {next_state.shape}")

    def log_model_state(self, model: torch.nn.Module, epoch: int):
        """Log model state for debugging."""
        # Log model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"parameters/{name}", param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)

    def log_memory_state(self, memory_size: int, memory_capacity: int):
        """Log replay memory state."""
        self.logger.debug(f"Memory size: {memory_size}/{memory_capacity}")

    def log_domain_metrics(self, domain: str, metrics: Dict[str, float]):
        """Log domain-specific metrics."""
        self.logger.info(f"\n{domain} Domain Metrics:")
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
            self.writer.add_scalar(f"domain/{domain}/{metric_name}", value)

    def log_error(self, error: Exception, context: str = ""):
        """Log error with context."""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)

    def log_warning(self, message: str, context: str = ""):
        """Log warning with context."""
        self.logger.warning(f"Warning in {context}: {message}")

    def save_metrics(self):
        """Save metrics to file."""
        metrics_path = os.path.join(self.run_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

    def _save_config(self):
        """Save configuration to file."""

        def _serialize_config(config):
            result = {}
            for k, v in config.__dict__.items():
                try:
                    json.dumps(v)
                    result[k] = v
                except TypeError:
                    result[k] = str(v)
            return result

        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(_serialize_config(self.config), f, indent=4)

    def close(self):
        """Close logger."""
        self.writer.close()
        self.save_metrics()
