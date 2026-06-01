"""
Multi-Domain Training (MDT) implementation for STYLE.
"""

import random
import torch
import numpy as np
import math
import time
from collections import deque
from typing import List, Dict, Any, Optional, Tuple, Union
from ..models.disp import DISP
from ..models.retriever import Retriever
from ..utils.llm_integration import LLMIntegration
from ..utils.monitoring import Monitor
from ..config import Config
from ..data.dataset_manager import DatasetManager
from ..data.dataset_manager_full import ConversationDataset, custom_collate_fn
from torch import optim
import wandb
import torch.nn.functional as F
from ..utils.replay_memory import ReplayMemory
import os
import glob
from torch.serialization import add_safe_globals
from collections import defaultdict
from torch.utils.data import DataLoader
from ..utils import TrainingLogger
from ..utils.training_monitor import TrainingStabilityMonitor
import json

# Add ReplayMemory to safe globals for loading
add_safe_globals([ReplayMemory])

__all__ = ["MDTTrainer", "ReplayMemory"]


def format_time(seconds):
    """Format seconds into H:M:S."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"


class MDTTrainer:
    """Multi-Domain Training for DISP model."""

    def __init__(self, config: Config):
        """Initialize the trainer."""
        self.config = config
        self.device = config.DEVICE

        # Allow externally provided data
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Initialize wandb
        try:
            wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                config=vars(config),
                name=f"STYLE_DISP_{config.DOMAINS}",
            )
            print("✅ Weights & Biases initialized successfully")
        except Exception as e:
            print(f"⚠️ Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")

        # Initialize DISP model
        self.disp_model = DISP(config)

        # Initialize learning rate scheduler for stability
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.disp_model.optimizer,
            mode="min",
            factor=0.8,  # Less aggressive decay (was 0.5)
            patience=10,  # Increased patience (was 5)
            min_lr=1e-6,  # Set minimum learning rate
        )

        # Initialize dataset manager
        self.dataset_manager = DatasetManager()

        # Load document texts
        doc_texts_path = os.path.join(config.DATA_DIR, "document_texts.json")
        if os.path.exists(doc_texts_path):
            print(f"Loading document texts from {doc_texts_path}")
            self.dataset_manager.load_document_texts(doc_texts_path)
        else:
            print(f"Warning: Document texts file not found at {doc_texts_path}")

        # Initialize retriever
        self.retriever = Retriever(config)

        # Initialize LLM integration
        self.llm = LLMIntegration(config)

        # Initialize logger
        self.logger = TrainingLogger(config)

        # Initialize training log file
        self.log_file_path = os.path.join(
            config.LOG_DIR, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        os.makedirs(config.LOG_DIR, exist_ok=True)

        # Initialize metrics
        self.metrics = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list),
        }

        # Initialize domain metrics for strategy diversity
        self.domain_metrics = defaultdict(list)

        # Initialize best model tracking
        self.best_val_reward = float("-inf")
        self.best_model_state = None

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_distribution = defaultdict(int)

        # Action sequence tracking for strategy diversity
        self.current_episode_actions = []
        self.episode_action_sequences = []

        # Training stability tracking
        self.loss_history = []
        self.gradient_norms = []
        self.learning_rate_history = []

        # Initialize stability monitor
        self.stability_monitor = TrainingStabilityMonitor(window_size=100)

        # Load documents for each domain
        for domain in config.DOMAINS:
            self.load_documents_for_domain(domain)

    def train(self, num_epochs: int):
        """Train the model for specified number of epochs."""
        try:
            print("\n🚀 Starting training...")
            self._write_log("🚀 Starting STYLE DISP training...")

            # Enable gradient anomaly detection for debugging
            torch.autograd.set_detect_anomaly(True)

            # Initialize ETA tracking
            step_times = deque(maxlen=50)  # for moving average
            start_time = time.time()

            # Use externally provided data if available
            if self.train_data is not None and self.val_data is not None and self.test_data is not None:
                print("\n📂 Using externally provided train/val/test data (e.g., diagnosis prompts CSVs)...")
                train_data = self.train_data
                val_data = self.val_data
                test_data = self.test_data
            else:
                # Load data for both domains
                train_data = []
                val_data = []
                test_data = []

                for domain in ["clariq", "opendialkg"]:
                    print(f"\n📂 Loading {domain} data...")
                    train_data.extend(self._load_data_file("train", domain))
                    val_data.extend(self._load_data_file("val", domain))
                    test_data.extend(self._load_data_file("test", domain))

            print(f"\n📊 Loaded data:")
            print(f"Train samples: {len(train_data)}")
            print(f"Val samples: {len(val_data)}")
            print(f"Test samples: {len(test_data)}")

            # Log data loading
            data_message = f"Data loaded: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}"
            self._write_log(data_message)

            # Store test data for evaluation
            self.test_data = test_data

            # Calculate total steps for ETA
            total_steps = len(train_data) * num_epochs
            global_step = 0

            # Create checkpoint directory
            os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)

            for epoch in range(num_epochs):
                print(f"\n🔄 Epoch {epoch + 1}/{num_epochs}")

                # Training phase
                print("🎯 Training phase...")
                train_metrics = self._train_epoch_with_eta(
                    train_data, epoch, step_times, global_step, total_steps
                )
                self._update_metrics("train", train_metrics)

                # Update global step count
                global_step += len(train_data)

                # Print training progress
                print("\n📈 Training Progress:")
                print(f"Loss: {train_metrics.get('loss', 0):.4f}")
                print(f"Reward: {train_metrics.get('reward', 0):.4f}")
                print(f"Success Rate: {train_metrics.get('success_rate', 0):.4f}")
                print(
                    f"Strategy Diversity: {train_metrics.get('strategy_diversity', 0):.4f}"
                )
                print(f"Average Turns: {train_metrics.get('avg_turns', 0):.2f}")

                # Enhanced stability monitoring
                if self.loss_history:
                    recent_losses = (
                        self.loss_history[-10:]
                        if len(self.loss_history) >= 10
                        else self.loss_history
                    )
                    avg_recent_loss = sum(recent_losses) / len(recent_losses)
                    loss_trend = (
                        "↗️" if avg_recent_loss > train_metrics.get("loss", 0) else "↘️"
                    )
                    print(
                        f"Loss Trend: {loss_trend} (Recent avg: {avg_recent_loss:.4f})"
                    )

                if self.learning_rate_history:
                    current_lr = self.learning_rate_history[-1]
                    print(f"Learning Rate: {current_lr:.6f}")

                # Compute and log overall strategy diversity
                overall_diversity = self._compute_overall_strategy_diversity()
                print(f"Overall Strategy Diversity: {overall_diversity:.4f}")
                print(f"Total Episodes: {len(self.episode_action_sequences)}")

                # Print action distribution with enhanced monitoring
                action_probs = [
                    train_metrics.get(f"action_{i}_prob", 0) for i in range(3)
                ]
                print(
                    f"Action Distribution: Ask={action_probs[0]:.3f}, Clarify={action_probs[1]:.3f}, Answer={action_probs[2]:.3f}"
                )

                # Enhanced stability warnings with specific recommendations
                if train_metrics.get("loss", 0) > 3.0:
                    print(
                        "⚠️ Warning: High loss detected. Consider reducing learning rate or increasing batch size."
                    )
                if overall_diversity < 0.7:
                    print(
                        "⚠️ Warning: Low strategy diversity. Model may be stuck in one action. Entropy regularization active."
                    )
                if (
                    action_probs[0] > 0.8
                    or action_probs[1] > 0.8
                    or action_probs[2] > 0.8
                ):
                    print(
                        "⚠️ Warning: Action distribution is skewed. Consider adjusting epsilon or reward structure."
                    )

                # Learning rate warnings
                if self.learning_rate_history and self.learning_rate_history[-1] < 1e-5:
                    print(
                        "⚠️ Warning: Learning rate very low. Consider resetting scheduler or increasing patience."
                    )

                # Gradient norm warnings
                if self.gradient_norms and self.gradient_norms[-1] > 2.0:
                    print(
                        "⚠️ Warning: High gradient norm detected. Gradient clipping may be needed."
                    )

                # Generate stability report every 5 epochs
                if (epoch + 1) % 5 == 0:
                    stability_report = self.stability_monitor.generate_report()
                    print(f"\n📊 Stability Report (Epoch {epoch + 1}):")
                    print(stability_report)

                    # Save stability plot
                    plot_path = os.path.join(
                        self.config.LOG_DIR, f"stability_plot_epoch_{epoch + 1}.png"
                    )
                    self.stability_monitor.plot_metrics(save_path=plot_path)

                # Validation phase
                print("\n🔍 Validation phase...")
                val_metrics = self._validate_epoch(val_data, epoch)
                self._update_metrics("val", val_metrics)

                # Print validation progress
                print("\n📊 Validation Progress:")
                print(f"Reward: {val_metrics.get('reward', 0):.4f}")
                print(f"Success Rate: {val_metrics.get('success_rate', 0):.4f}")
                print(
                    f"Strategy Diversity: {val_metrics.get('strategy_diversity', 0):.4f}"
                )
                print(f"Average Turns: {val_metrics.get('avg_turns', 0):.2f}")

                # Print validation action distribution
                val_action_probs = [
                    val_metrics.get(f"action_{i}_prob", 0) for i in range(3)
                ]
                print(
                    f"Action Distribution: Ask={val_action_probs[0]:.3f}, Clarify={val_action_probs[1]:.3f}, Answer={val_action_probs[2]:.3f}"
                )

                # Log to wandb
                try:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train/loss": train_metrics.get("loss", 0),
                            "train/reward": train_metrics.get("reward", 0),
                            "train/success_rate": train_metrics.get("success_rate", 0),
                            "train/strategy_diversity": train_metrics.get(
                                "strategy_diversity", 0
                            ),
                            "train/overall_strategy_diversity": overall_diversity,
                            "train/avg_turns": train_metrics.get("avg_turns", 0),
                            "train/total_episodes": len(self.episode_action_sequences),
                            "val/reward": val_metrics.get("reward", 0),
                            "val/success_rate": val_metrics.get("success_rate", 0),
                            "val/strategy_diversity": val_metrics.get(
                                "strategy_diversity", 0
                            ),
                            "val/avg_turns": val_metrics.get("avg_turns", 0),
                            "epsilon": self.config.EPSILON,
                            "learning_rate": (
                                self.learning_rate_history[-1]
                                if self.learning_rate_history
                                else self.config.LEARNING_RATE
                            ),
                            "gradient_norm": (
                                self.gradient_norms[-1] if self.gradient_norms else 0.0
                            ),
                            "entropy": train_metrics.get("entropy", 0),
                            "action_ask_prob": action_probs[0],
                            "action_clarify_prob": action_probs[1],
                            "action_answer_prob": action_probs[2],
                        }
                    )
                except Exception as e:
                    print(f"⚠️ wandb.log failed: {e}")

                # Save best model
                if val_metrics["reward"] > self.best_val_reward:
                    self.best_val_reward = val_metrics["reward"]
                    self._save_model("best_model.pt")
                    best_model_message = (
                        f"New best model saved! Reward: {self.best_val_reward:.4f}"
                    )
                    print(f"\n🏆 {best_model_message}")
                    self._write_log(best_model_message)

                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(
                        epoch + 1, global_step, train_metrics, val_metrics
                    )
                    checkpoint_message = f"Checkpoint saved at epoch {epoch + 1}"
                    print(f"\n💾 {checkpoint_message}")
                    self._write_log(checkpoint_message)

                # Evaluate on test set every 50 epochs
                if (epoch + 1) % 50 == 0:
                    print("\n🧪 Evaluating on test set...")
                    eval_metrics = self.evaluate()
                    print("\n📋 Test Evaluation Metrics:")
                    for metric, value in eval_metrics.items():
                        print(f"{metric}: {value:.4f}")

                    # Log test metrics to wandb
                    try:
                        test_log = {"epoch": epoch}
                        for metric, value in eval_metrics.items():
                            test_log[f"test/{metric}"] = value
                        wandb.log(test_log)
                    except Exception as e:
                        print(f"⚠️ wandb.log failed: {e}")

            # Calculate total training time
            total_time = time.time() - start_time
            total_time_message = f"Total training time: {format_time(total_time)}"
            print(f"\n⏱️ {total_time_message}")
            self._write_log(total_time_message)

            # Final evaluation
            print("\n🎯 Final evaluation on test set...")
            final_eval_metrics = self.evaluate()
            print("\n🏁 Final Test Metrics:")
            for metric, value in final_eval_metrics.items():
                print(f"{metric}: {value:.4f}")

            # Log final metrics
            final_metrics_message = "Final Test Metrics: " + ", ".join(
                [f"{k}={v:.4f}" for k, v in final_eval_metrics.items()]
            )
            self._write_log(final_metrics_message)

            # Save final model
            self._save_model("final_model.pt")
            final_model_message = "Final model saved!"
            print(f"\n💾 {final_model_message}")
            self._write_log(final_model_message)

        except Exception as e:
            print(f"❌ Error during training: {e}")
            raise
        finally:
            print("\n✅ Training completed!")
            try:
                wandb.finish()
            except Exception as e:
                print(f"⚠️ Failed to finish wandb: {e}")

    def _load_data_file(
        self, split: str, domain: str = "clariq"
    ) -> List[Dict[str, Any]]:
        """Load data from file.

        Args:
            split: Data split ('train', 'val', or 'test')
            domain: Dataset domain ('clariq' or 'opendialkg')
        """
        # Map split names to actual filenames
        split_map = {
            "train": f"{domain}_train.json",
            "val": f"{domain}_dev.json",
            "test": f"{domain}_test.json",
        }

        data_path = os.path.join("data", split_map[split])
        try:
            with open(data_path, "r") as f:
                data = json.load(f)

            # Clean data by removing NaN values
            cleaned_data = []
            for item in data:
                cleaned_item = {}
                for key, value in item.items():
                    if key == "query_history" and isinstance(value, list):
                        # Clean query history
                        cleaned_queries = []
                        for query in value:
                            if (
                                isinstance(query, str)
                                and query.strip()
                                and query.lower() != "nan"
                            ):
                                cleaned_queries.append(query.strip())
                        cleaned_item[key] = (
                            cleaned_queries
                            if cleaned_queries
                            else ["What information are you looking for?"]
                        )
                    elif key == "documents" and isinstance(value, list):
                        # Clean documents
                        cleaned_docs = []
                        for doc in value:
                            if (
                                isinstance(doc, str)
                                and doc.strip()
                                and doc.lower() != "nan"
                            ):
                                cleaned_docs.append(doc.strip())
                        cleaned_item[key] = (
                            cleaned_docs
                            if cleaned_docs
                            else ["This document contains general information."]
                        )
                    elif key == "retrieval_scores" and isinstance(value, list):
                        # Clean retrieval scores
                        cleaned_scores = []
                        for score in value:
                            if isinstance(score, (int, float)) and not math.isnan(
                                score
                            ):
                                cleaned_scores.append(float(score))
                        cleaned_item[key] = cleaned_scores if cleaned_scores else [1.0]
                    else:
                        # Keep other fields as is
                        cleaned_item[key] = value

                # Only add items that have valid data
                if cleaned_item.get("query_history") and cleaned_item.get("documents"):
                    cleaned_data.append(cleaned_item)

            print(
                f"Loaded {len(cleaned_data)} cleaned samples from {data_path} (original: {len(data)})"
            )
            return cleaned_data
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            return []

    def _write_log(self, message: str):
        """Write a message to the training log file."""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            with open(self.log_file_path, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"⚠️ Failed to write to log file: {e}")

    def _reset_episode_actions(self):
        """Reset the current episode action sequence."""
        if self.current_episode_actions:
            # Store completed episode sequence
            self.episode_action_sequences.append(self.current_episode_actions.copy())
            # Reset for next episode
            self.current_episode_actions = []

    def _train_epoch_with_eta(
        self,
        train_data: List[Dict[str, Any]],
        epoch: int,
        step_times: deque,
        global_step: int,
        total_steps: int,
    ) -> Dict[str, float]:
        """Train for one epoch with ETA tracking."""
        self.disp_model.train()
        metrics = defaultdict(float)
        num_batches = 0

        # Reset episode actions at start of epoch
        self._reset_episode_actions()

        # Track additional metrics
        total_turns = 0
        successful_episodes = 0
        action_counts = defaultdict(int)

        # ETA tracking variables
        epoch_start_time = time.time()
        current_step = global_step

        for batch_idx, batch in enumerate(train_data):
            try:
                step_start_time = time.time()

                # Ensure domain is set
                if "domain" not in batch:
                    batch["domain"] = "clariq"  # Default domain

                # Get current state
                state = self._prepare_state(batch)

                # Get action from DISP
                action = self.disp_model.select_action(state)

                # Debug print for action selection
                if self.config.DEBUG:
                    print(f"🎯 Action Selection Debug:")
                    print(f"  Epsilon: {self.config.EPSILON:.4f}")
                    print(f"  Selected action: {action}")
                    print(f"  Action type: {type(action)}")
                    if isinstance(action, torch.Tensor):
                        print(f"  Action device: {action.device}")

                # Track action distribution
                if isinstance(action, torch.Tensor):
                    action_val = action.item()
                else:
                    action_val = action
                action_counts[action_val] += 1

                # Execute action
                next_state, reward, done, info = self._execute_action(
                    action=action,
                    domain=batch["domain"],
                    target_doc=self._get_target_doc(batch),
                )

                # Ensure all tensors are on the same device
                state = state.to(self.device)
                next_state = next_state.to(self.device)

                # Convert action to tensor if it's not already
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(
                        [action], device=self.device, dtype=torch.long
                    )
                else:
                    action = action.to(self.device)

                # Store transition in replay buffer
                self.disp_model.replay_buffer.push(
                    state, action, reward, next_state, done
                )

                # Train DISP
                loss = self.disp_model.train_on_batch()

                # Track loss history for stability monitoring
                if loss is not None:
                    self.loss_history.append(loss)

                    # Update learning rate scheduler with less aggressive decay
                    if (
                        len(self.loss_history) >= 20
                    ):  # Increased from 10 for more stability
                        recent_losses = self.loss_history[-20:]
                        avg_recent_loss = sum(recent_losses) / len(recent_losses)
                        self.scheduler.step(avg_recent_loss)

                    # Track current learning rate
                    current_lr = self.disp_model.optimizer.param_groups[0]["lr"]
                    self.learning_rate_history.append(current_lr)

                    # Entropy regularization is not part of the paper reward; track only in metrics.
                    with torch.no_grad():
                        q_values = self.disp_model.dqn(state)
                        entropy_reg = self._compute_entropy_regularization(q_values)
                        entropy_bonus = 0.0

                # Update metrics
                if loss is not None:
                    metrics["loss"] += loss
                    if self.config.DEBUG:
                        print(f"[TRAIN] Loss: {loss:.4f}")

                    # Update stability monitor with enhanced metrics
                    batch_metrics = {
                        "loss": loss,
                        "reward": reward,
                        "strategy_diversity": info["strategy_diversity"],
                        "gradient_norm": self._compute_gradient_norm(),
                        "learning_rate": (
                            current_lr
                            if loss is not None
                            else self.config.LEARNING_RATE
                        ),
                        "entropy": entropy_reg.item() if loss is not None else 0.0,
                    }
                    self.stability_monitor.update(batch_metrics)

                    try:
                        wandb.log(
                            {
                                "train/batch_loss": loss,
                                "train/batch_reward": reward,
                                "train/batch_success": info["success"],
                                "train/batch_strategy_diversity": info[
                                    "strategy_diversity"
                                ],
                                "train/learning_rate": (
                                    current_lr
                                    if loss is not None
                                    else self.config.LEARNING_RATE
                                ),
                                "train/epsilon": self.config.EPSILON,
                                "train/gradient_norm": (
                                    self._compute_gradient_norm()
                                    if loss is not None
                                    else 0.0
                                ),
                                "train/entropy": (
                                    entropy_reg.item() if loss is not None else 0.0
                                ),
                                "train/entropy_bonus": (
                                    entropy_bonus if loss is not None else 0.0
                                ),
                            }
                        )
                    except Exception as e:
                        if self.config.DEBUG:
                            print(f"[WARN] wandb.log failed: {e}")

                metrics["reward"] += reward
                metrics["success_rate"] += float(info["success"])
                metrics["strategy_diversity"] += info["strategy_diversity"]

                # Track additional metrics
                total_turns += info.get("queries", 1)
                if info["success"]:
                    successful_episodes += 1

                num_batches += 1
                current_step += 1

                # Calculate ETA
                step_duration = time.time() - step_start_time
                step_times.append(step_duration)

                if len(step_times) > 0:
                    avg_step_time = sum(step_times) / len(step_times)
                    steps_left = total_steps - current_step
                    eta_seconds = avg_step_time * steps_left
                    eta_time = format_time(eta_seconds)

                    # Progress update every 10 batches or if debug mode
                    if batch_idx % 10 == 0 or self.config.DEBUG:
                        progress = f"[{current_step}/{total_steps}] "
                        loss_info = (
                            f"Loss: {loss:.4f}" if loss is not None else "Loss: N/A"
                        )
                        eta_info = f"ETA: {eta_time}"
                        step_time_info = f"Step: {step_duration:.2f}s"

                        progress_message = (
                            f"{progress}{loss_info} | {eta_info} | {step_time_info}"
                        )
                        print(progress_message, end="\r", flush=True)

                        # Write to log file
                        self._write_log(f"Step {current_step}: {progress_message}")

                    # Save checkpoint every 1000 steps
                    if batch_idx % 1000 == 0 and batch_idx > 0:
                        checkpoint_path = os.path.join(
                            self.config.CHECKPOINT_DIR,
                            f"checkpoint_step_{current_step}.pt",
                        )
                        torch.save(
                            {
                                "model_state_dict": self.disp_model.state_dict(),
                                "optimizer_state_dict": self.disp_model.optimizer.state_dict(),
                                "step": current_step,
                                "epoch": epoch,
                                "batch_idx": batch_idx,
                                "epsilon": self.config.EPSILON,
                                "metrics": dict(metrics),
                            },
                            checkpoint_path,
                        )
                        checkpoint_message = f"Checkpoint saved at step {current_step} → {checkpoint_path}"
                        print(f"\n✅ {checkpoint_message}")
                        self._write_log(checkpoint_message)

            except Exception as e:
                print(f"Error in batch: {e}")
                if self.config.DEBUG:
                    print(f"Batch contents: {batch}")
                continue

        # Print final progress for this epoch
        epoch_duration = time.time() - epoch_start_time
        epoch_message = f"Epoch {epoch + 1} completed in {format_time(epoch_duration)}"
        print(f"\n⏱️ {epoch_message}")
        self._write_log(epoch_message)

        # Average metrics
        if num_batches > 0:
            for key in metrics:
                metrics[key] /= num_batches

            # Add additional metrics
            metrics["avg_turns"] = total_turns / num_batches
            metrics["success_rate"] = successful_episodes / num_batches

            # Log action distribution
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                for action, count in action_counts.items():
                    action_prob = count / total_actions
                    metrics[f"action_{action}_prob"] = action_prob
                    try:
                        wandb.log({f"train/action_{action}_probability": action_prob})
                    except Exception as e:
                        if self.config.DEBUG:
                            print(f"[WARN] wandb.log failed: {e}")

            # Log epoch metrics
            metrics_message = f"Epoch {epoch + 1} metrics: Loss={metrics.get('loss', 0):.4f}, Reward={metrics.get('reward', 0):.4f}, Success={metrics.get('success_rate', 0):.4f}"
            self._write_log(metrics_message)

        return dict(metrics)

    def _validate_epoch(
        self, val_data: List[Dict[str, Any]], epoch: int
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.disp_model.eval()
        metrics = defaultdict(float)
        num_batches = 0

        # Reset episode actions at start of validation
        self._reset_episode_actions()

        # Track additional metrics
        total_turns = 0
        successful_episodes = 0
        action_counts = defaultdict(int)

        with torch.no_grad():
            for batch in val_data:
                try:
                    # Ensure domain is set
                    if "domain" not in batch:
                        batch["domain"] = "clariq"  # Default domain

                    # Get current state
                    state = self._prepare_state(batch)

                    # Get action from DISP
                    action = self.disp_model.select_action(state)

                    # Debug print for action selection
                    if self.config.DEBUG:
                        print(f"🎯 Action Selection Debug:")
                        print(f"  Epsilon: {self.config.EPSILON:.4f}")
                        print(f"  Selected action: {action}")
                        print(f"  Action type: {type(action)}")
                        if isinstance(action, torch.Tensor):
                            print(f"  Action device: {action.device}")

                    # Track action distribution
                    if isinstance(action, torch.Tensor):
                        action_val = action.item()
                    else:
                        action_val = action
                    action_counts[action_val] += 1

                    # Execute action
                    next_state, reward, done, info = self._execute_action(
                        action=action,
                        domain=batch["domain"],
                        target_doc=self._get_target_doc(batch),
                    )

                    # Update metrics
                    metrics["reward"] += reward
                    metrics["success_rate"] += float(info["success"])
                    metrics["strategy_diversity"] += info["strategy_diversity"]

                    # Track additional metrics
                    total_turns += info.get("queries", 1)
                    if info["success"]:
                        successful_episodes += 1

                    num_batches += 1

                except Exception as e:
                    print(f"Error in batch: {e}")
                    if self.config.DEBUG:
                        print(f"Batch contents: {batch}")
                    continue

        # Average metrics
        if num_batches > 0:
            for key in metrics:
                metrics[key] /= num_batches

            # Add additional metrics
            metrics["avg_turns"] = total_turns / num_batches
            metrics["success_rate"] = successful_episodes / num_batches

            # Log action distribution
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                for action, count in action_counts.items():
                    action_prob = count / total_actions
                    metrics[f"action_{action}_prob"] = action_prob
                    try:
                        wandb.log({f"val/action_{action}_probability": action_prob})
                    except Exception as e:
                        if self.config.DEBUG:
                            print(f"[WARN] wandb.log failed: {e}")

        return dict(metrics)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model with proper error handling."""
        self.disp_model.eval()
        metrics = defaultdict(float)
        num_batches = 0

        try:
            with torch.no_grad():
                for batch in self.test_data:
                    try:
                        # Get current state
                        state = self._prepare_state(batch)

                        # Get action from DISP - use eval_mode for greedy selection
                        action = self.disp_model.select_action(state, eval_mode=True)

                        # Execute action
                        next_state, reward, done, info = self._execute_action(
                            action=action,
                            domain=batch["domain"],
                            target_doc=self._get_target_doc(batch),
                        )

                        # Update metrics
                        metrics["reward"] += reward
                        metrics["success_rate"] += float(info["success"])
                        metrics["strategy_diversity"] += info["strategy_diversity"]

                        num_batches += 1

                    except Exception as e:
                        print(f"Error in batch: {e}")
                        print(f"Batch contents: {batch}")
                        continue

            # Average metrics
            if num_batches > 0:
                for key in metrics:
                    metrics[key] /= num_batches
                print(f"✅ Evaluation completed successfully on {num_batches} batches")
            else:
                print("⚠️ Warning: No successful evaluation batches")

        except Exception as e:
            print(f"❌ Critical error in evaluation: {e}")
            import traceback

            traceback.print_exc()

        return dict(metrics)

    def _execute_action(
        self, action: Union[int, torch.Tensor], domain: str, target_doc: str
    ) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Execute an action and return next state, reward, done flag, and info."""
        info = {
            "queries": 0,
            "responses": 0,
            "success": False,
            "strategy_diversity": 0.0,
        }

        try:
            if self.config.DEBUG:
                print(f"\nExecuting action: {action}")
                if isinstance(action, torch.Tensor):
                    print(f"Action shape: {action.shape}")

            if isinstance(action, torch.Tensor):
                if action.numel() == 1:
                    action = action.item()
                else:
                    action = action[0].item()  # Take first item if batch

            if self.config.DEBUG:
                print(f"Converted action: {action}, target doc: {target_doc}")

            self.current_episode_actions.append(action)

            if action == self.config.ACTION_ASK:  # ask clarification
                docs, scores = self.retriever.retrieve(target_doc, domain)

                if isinstance(scores, torch.Tensor):
                    if scores.dim() > 1:
                        scores = scores[0]
                    scores = scores.cpu().numpy().tolist()

                info["documents"] = docs

                try:
                    question = self.llm.generate_clarification_question(
                        query_history=[target_doc],
                        retrieved_docs=docs,
                        retrieval_scores=scores,
                    )
                except Exception as e:
                    if self.config.DEBUG:
                        print(f"LLM clarification failed: {e}")
                    question = "Could you please provide more details about what you're looking for?"

                try:
                    response = self.llm.simulate_user_response(
                        user_intent=target_doc, question=question
                    )
                except Exception as e:
                    if self.config.DEBUG:
                        print(f"LLM user simulation failed: {e}")
                    response = target_doc

                info["queries"] = 1
                info["responses"] = 1
                reward = self.config.INTERMEDIATE_REWARD
                done = False
                query_history = [target_doc, response]

            else:  # answer
                docs, scores = self.retriever.retrieve(target_doc, domain)

                if isinstance(scores, torch.Tensor):
                    if scores.dim() > 1:
                        scores = scores[0]
                    scores = scores.cpu().numpy().tolist()

                info["documents"] = docs
                target_rank = self._get_target_rank(target_doc, docs)
                success = 0 <= target_rank < self.config.TOP_K
                reward = (
                    self.config.SUCCESS_REWARD
                    if success
                    else self.config.REWARD_TIMEOUT
                )
                info["success"] = success
                info["queries"] = 1
                done = True
                query_history = [target_doc]

            # Calculate strategy diversity using current episode actions
            strategy_diversity = self._compute_strategy_diversity(
                self.current_episode_actions
            )
            info["strategy_diversity"] = strategy_diversity

            # Debug print strategy diversity
            if self.config.DEBUG:
                print(f"🔍 Strategy Diversity Debug:")
                print(f"  Current episode actions: {self.current_episode_actions}")
                print(f"  Strategy diversity: {strategy_diversity:.4f}")
                print(f"  Unique actions: {len(set(self.current_episode_actions))}")
                print(f"  Total actions: {len(self.current_episode_actions)}")

            next_state = self._prepare_state(
                {
                    "query_history": query_history,
                    "documents": docs,
                    "retrieval_scores": scores,
                }
            )

            return next_state, reward, done, info

        except Exception as e:
            print(f"Error executing action: {e}")
            print(f"Action that caused error: {action}")
            print(f"Action type: {type(action)}")
            if isinstance(action, torch.Tensor):
                print(f"Action shape: {action.shape}")
                print(f"Action device: {action.device}")
            # Return default values in case of error
            default_state = self.disp_model.state_builder.zero_state()
            return default_state, self.config.REWARD_TIMEOUT, True, info

    def _update_metrics(self, split: str, metrics: Dict[str, float]):
        """Update metrics for a split."""
        for metric_name, value in metrics.items():
            self.metrics[split][metric_name].append(value)

    def _print_epoch_metrics(
        self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]
    ):
        """Print metrics for an epoch."""
        print("\nTraining Metrics:")
        for metric_name, value in train_metrics.items():
            print(f"{metric_name}: {value:.4f}")

        print("\nValidation Metrics:")
        for metric_name, value in val_metrics.items():
            print(f"{metric_name}: {value:.4f}")

    def _save_model(self, filename: str):
        """Save model state."""
        save_path = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        torch.save(self.disp_model.state_dict(), save_path)

    def _load_model(self, filename: str):
        """Load model state.

        Args:
            filename: Name of the checkpoint file to load
        """
        load_path = os.path.join(self.config.MODEL_SAVE_DIR, filename)

        try:
            checkpoint = torch.load(load_path, map_location=self.device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Full checkpoint format
                self.disp_model.load_state_dict(checkpoint["model_state_dict"])
                print(f"✅ Successfully loaded full checkpoint from {load_path}")

                # Optionally load other checkpoint components
                if "optimizer_state_dict" in checkpoint:
                    self.disp_model.optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )
                    print(f"✅ Loaded optimizer state")

                if "epoch" in checkpoint:
                    print(f"✅ Checkpoint from epoch {checkpoint['epoch']}")

                if "epsilon" in checkpoint:
                    self.config.EPSILON = checkpoint["epsilon"]
                    print(f"✅ Loaded epsilon: {checkpoint['epsilon']}")

            else:
                # Raw state_dict format
                self.disp_model.load_state_dict(checkpoint)
                print(f"✅ Loaded raw model weights from {load_path}")

        except Exception as e:
            print(f"❌ Error loading checkpoint from {load_path}: {e}")
            print(f"Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
            raise

    def load_documents_for_domain(self, domain: str):
        """Load documents for a specific domain."""
        domain = domain.lower()  # Convert to lowercase for consistency
        if domain == "clariq":
            # Load ClariQ documents
            documents = [
                "What is the capital of France?",
                "How do I make chocolate cake?",
                "What is the weather like in New York?",
                "How to fix a flat tire?",
                "What is the meaning of life?",
                "How to learn Python programming?",
                "What are the symptoms of COVID-19?",
                "How to train a dog?",
                "What is machine learning?",
                "How to cook pasta?",
                "What is quantum computing?",
                "How to write a resume?",
                "What is the best way to learn a new language?",
                "How to start a business?",
                "What is blockchain technology?",
            ]
        elif domain == "opendialkg":
            # Load OpenDialKG documents
            documents = [
                "How to install Windows 10?",
                "How to fix blue screen error?",
                "How to update graphics drivers?",
                "How to recover deleted files?",
                "How to speed up my computer?",
                "How to install Python?",
                "How to reset Windows password?",
                "How to uninstall programs?",
                "How to check system requirements?",
                "How to update drivers?",
                "How to fix blue screen errors?",
                "How to backup files?",
                "How to clean up disk space?",
                "How to connect to WiFi?",
                "How to troubleshoot network issues?",
            ]
        else:
            raise ValueError(f"Unknown domain: {domain}")

        # Load documents into retriever
        self.retriever.load_documents(domain=domain, documents=documents)

    def _load_documents(self, domain):
        """Load documents for the given domain."""
        if domain == "clariq":
            # Sample ClariQ documents
            return [
                "What is the capital of France?",
                "How do I make chocolate cake?",
                "What are the symptoms of COVID-19?",
                "How to learn Python programming?",
                "What is machine learning?",
                "How to cook pasta?",
                "What is the meaning of life?",
                "How to fix a flat tire?",
                "What is quantum computing?",
                "How to train a dog?",
            ]
        elif domain == "opendialkg":
            # Sample OpenDialKG documents
            return [
                "How to fix Windows update issues?",
                "How to install Python?",
                "How to reset Windows password?",
                "How to uninstall programs?",
                "How to check system requirements?",
                "How to update drivers?",
                "How to fix blue screen errors?",
                "How to backup files?",
                "How to clean up disk space?",
                "How to connect to WiFi?",
            ]
        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": self.disp_model.state_dict(),
            "optimizer_state_dict": self.disp_model.optimizer.state_dict(),
            "replay_buffer": self.disp_model.replay_buffer,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "epsilon": self.config.EPSILON,
            "best_val_reward": self.best_val_reward,
            "episode_action_sequences": self.episode_action_sequences,
            "action_distribution": self.action_distribution,
        }
        path = os.path.join(self.config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch}")

        # Keep only the last 3 checkpoints
        checkpoints = sorted(
            glob.glob(os.path.join(self.config.CHECKPOINT_DIR, "checkpoint_*.pt"))
        )
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load a training checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Dictionary containing checkpoint information
        """
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle both full checkpoint and raw state_dict formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Full checkpoint format
                print(f"✅ Loading full checkpoint format")

                # Load model state
                self.disp_model.load_state_dict(checkpoint["model_state_dict"])

                # Load optimizer state
                if "optimizer_state_dict" in checkpoint:
                    self.disp_model.optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )

                # Load replay buffer
                if "replay_buffer" in checkpoint:
                    self.disp_model.replay_buffer = checkpoint["replay_buffer"]

                # Load training state
                if "epsilon" in checkpoint:
                    self.config.EPSILON = checkpoint["epsilon"]

                if "best_val_reward" in checkpoint:
                    self.best_val_reward = checkpoint["best_val_reward"]

                if "episode_action_sequences" in checkpoint:
                    self.episode_action_sequences = checkpoint[
                        "episode_action_sequences"
                    ]

                if "action_distribution" in checkpoint:
                    self.action_distribution = checkpoint["action_distribution"]

            else:
                # Raw state_dict format
                print(f"✅ Loading raw state_dict format")
                self.disp_model.load_state_dict(checkpoint)

                # Set default values for missing components
                print(f"⚠️ Using default values for optimizer, epsilon, etc.")

            print(f"✅ Checkpoint loaded successfully")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
            print(f"  Epsilon: {checkpoint.get('epsilon', self.config.EPSILON)}")
            print(
                f"  Best validation reward: {checkpoint.get('best_val_reward', 'N/A')}"
            )

            return checkpoint

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print(
                f"Checkpoint type: {type(checkpoint) if 'checkpoint' in locals() else 'Unknown'}"
            )
            if "checkpoint" in locals() and isinstance(checkpoint, dict):
                print(f"Checkpoint keys: {list(checkpoint.keys())}")
            return {}

    def resume_training(self, checkpoint_path: str, num_epochs: int) -> None:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
            num_epochs: Number of additional epochs to train
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        if not checkpoint:
            print("❌ Failed to load checkpoint, starting fresh training")
            self.train(num_epochs)
            return

        # Calculate remaining epochs
        current_epoch = checkpoint.get("epoch", 0)
        remaining_epochs = num_epochs - current_epoch

        if remaining_epochs <= 0:
            print(
                f"✅ Training already completed ({current_epoch}/{num_epochs} epochs)"
            )
            return

        print(f"🔄 Resuming training for {remaining_epochs} more epochs")
        self.train(remaining_epochs)

    def train_on_batch(self) -> Optional[float]:
        """Train on a single batch of data."""
        try:
            # Sample batch
            batch = self._sample_batch()

            # Debug prints
            print("\nBatch contents:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"{key}: {type(value)}")

            # Get current state
            state = self._prepare_state(batch)
            print(f"State shape: {state.shape}")

            # Get action from DISP - only pass state tensor
            action = self.disp_model.select_action(state)

            # Execute action
            next_state, reward, done, info = self._execute_action(
                action=action,
                domain=batch["domain"],
                target_doc=self._get_target_doc(batch),
            )

            # Store transition in replay buffer
            self.disp_model.memory.push(state, action, next_state, reward, done)

            # Train DISP
            loss = self.disp_model.train_on_batch()

            # Track loss history for stability monitoring
            if loss is not None:
                self.loss_history.append(loss)

                # Update learning rate scheduler with less aggressive decay
                if len(self.loss_history) >= 20:  # Increased from 10 for more stability
                    recent_losses = self.loss_history[-20:]
                    avg_recent_loss = sum(recent_losses) / len(recent_losses)
                    self.scheduler.step(avg_recent_loss)

                # Track current learning rate
                current_lr = self.disp_model.optimizer.param_groups[0]["lr"]
                self.learning_rate_history.append(current_lr)

                # Add entropy regularization to encourage diversity
                with torch.no_grad():
                    q_values = self.disp_model.dqn(state)
                    entropy_reg = self._compute_entropy_regularization(q_values)
                    # Add small entropy bonus to reward
                    entropy_bonus = 0.01 * entropy_reg.item()  # Small regularization
                    reward += entropy_bonus

            # Update metrics
            self._update_metrics(
                "train",
                {
                    "loss": loss,
                    "reward": reward,
                    "success": info["success"],
                    "strategy_diversity": info["strategy_diversity"],
                },
            )

            return loss

        except Exception as e:
            print(f"Error in batch: {e}")
            print(f"Batch contents: {batch}")
            return None

    def _save_local_metrics(self, metrics: Dict[str, float]) -> None:
        """Save metrics locally when wandb logging fails."""
        # Create metrics directory if it doesn't exist
        metrics_dir = "local_metrics"
        os.makedirs(metrics_dir, exist_ok=True)

        # Save metrics with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metrics_dir}/metrics_{timestamp}.json"

        try:
            with open(filename, "w") as f:
                json.dump(metrics, f)
            print(f"Metrics saved locally to {filename}")
        except Exception as e:
            print(f"Failed to save local metrics: {str(e)}")

    def save_model(self, path: str) -> None:
        """Save the final trained model.

        Args:
            path: Path to save the model to
        """
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(path)
        if save_dir:  # Only create directory if path contains a directory
            os.makedirs(save_dir, exist_ok=True)

        # Save model state
        torch.save(
            {
                "model_state": self.disp_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "metrics": self.metrics,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "action_distribution": self.action_distribution,
                "domain_metrics": self.domain_metrics,
            },
            path,
        )
        print(f"Model saved to {path}")

        # Also save to wandb
        try:
            wandb.save(path)
        except Exception as e:
            print(f"Warning: Failed to save model to wandb: {str(e)}")

    def _prepare_state(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Prepare state tensor via DISP (full history + k docs + k scores)."""
        return self.disp_model._prepare_state(batch)

    def _get_target_rank(self, target_doc: str, retrieved_docs: List[str]) -> int:
        """Get the rank of the target document using BERT embeddings."""
        try:
            if not target_doc or not retrieved_docs:
                return -1

            # Use the BERT encoder's high-level interface
            bert_output = self.disp_model.bert_encoder(
                query_history=[target_doc], documents=retrieved_docs
            )

            # Get target embedding (first document)
            target_embedding = bert_output["doc_embeddings"][:, 0, :]  # [1, bert_dim]

            # Get retrieved document embeddings
            doc_embeddings = bert_output["doc_embeddings"]  # [1, num_docs, bert_dim]

            # Compute cosine similarity
            similarities = F.cosine_similarity(
                target_embedding.unsqueeze(1),
                doc_embeddings.squeeze(0).unsqueeze(0),
                dim=2,
            ).squeeze()

            # Get rank of target document (assuming it's the first one)
            target_rank = (similarities > similarities[0]).sum().item()

            return target_rank

        except Exception as e:
            print(f"Error getting target rank: {e}")
            return -1

    def _sample_batch(self) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences from the replay buffer."""
        batch = random.sample(self.disp_model.replay_buffer, self.config.BATCH_SIZE)

        return {
            "state": torch.stack([exp["state"] for exp in batch]),
            "action": torch.stack([exp["action"] for exp in batch]),
            "reward": torch.stack([exp["reward"] for exp in batch]),
            "next_state": torch.stack([exp["next_state"] for exp in batch]),
            "done": torch.stack([exp["done"] for exp in batch]),
        }

    def _compute_strategy_diversity(self, action_sequence: List[int]) -> float:
        """Compute strategy diversity using a simple entropy-based approach.

        Args:
            action_sequence: List of action indices

        Returns:
            Strategy diversity score between 0 and 1
        """
        if not action_sequence:
            return 0.0

        # Convert to list if it's a tensor
        if isinstance(action_sequence, torch.Tensor):
            action_sequence = action_sequence.cpu().numpy().tolist()

        # Simple diversity: ratio of unique actions to total actions
        unique_actions = len(set(action_sequence))
        total_actions = len(action_sequence)

        if total_actions == 0:
            return 0.0

        # Normalize by number of possible actions (3: ask, clarify, answer)
        diversity = unique_actions / min(total_actions, 3)

        # Add entropy-based diversity for more nuanced measurement
        from collections import Counter

        action_counts = Counter(action_sequence)
        total = sum(action_counts.values())

        if total == 0:
            return diversity

        # Compute entropy
        entropy = 0.0
        for count in action_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)

        # Normalize entropy by maximum possible entropy (log2(3) for 3 actions)
        max_entropy = np.log2(3)
        entropy_diversity = entropy / max_entropy if max_entropy > 0 else 0.0

        # Combine both metrics with more weight on entropy
        combined_diversity = 0.3 * diversity + 0.7 * entropy_diversity

        return float(combined_diversity)

    def _compute_entropy_regularization(self, action_probs: torch.Tensor) -> float:
        """Compute entropy regularization to encourage exploration.

        Args:
            action_probs: Action probabilities [batch_size, num_actions]

        Returns:
            Entropy regularization term
        """
        # Compute entropy
        log_probs = torch.log_softmax(action_probs, dim=1)
        entropy = -torch.sum(torch.softmax(action_probs, dim=1) * log_probs, dim=1)

        # Return mean entropy across batch
        return entropy.mean()

    def _get_target_doc(self, batch: Dict[str, Any]) -> str:
        """Safely extract target document from batch.

        Args:
            batch: Dictionary containing batch data

        Returns:
            Target document string

        Raises:
            ValueError: If no target document can be found
        """
        # Try to get target_doc directly
        if "target_doc" in batch and batch["target_doc"]:
            target_doc = batch["target_doc"]
            if (
                isinstance(target_doc, str)
                and target_doc.strip()
                and target_doc.lower() != "nan"
            ):
                return target_doc

        # Fallback to first document in documents list
        if "documents" in batch and batch["documents"]:
            for doc in batch["documents"]:
                if isinstance(doc, str) and doc.strip() and doc.lower() != "nan":
                    return doc

        # Fallback to first query in query_history
        if "query_history" in batch and batch["query_history"]:
            # Filter out NaN values
            for q in batch["query_history"]:
                if isinstance(q, str) and q.strip() and q.lower() != "nan":
                    return q

        # Final fallback
        return "What information are you looking for?"

    def _compute_overall_strategy_diversity(self) -> float:
        """Compute strategy diversity across all episode sequences."""
        if not self.episode_action_sequences:
            return 0.0

        # Compute diversity for each episode
        episode_diversities = []
        for sequence in self.episode_action_sequences:
            diversity = self._compute_strategy_diversity(sequence)
            episode_diversities.append(diversity)

        # Return average diversity across episodes
        return (
            sum(episode_diversities) / len(episode_diversities)
            if episode_diversities
            else 0.0
        )

    def _compute_gradient_norm(self) -> float:
        """Compute the L2 norm of gradients for monitoring training stability."""
        total_norm = 0.0
        for p in self.disp_model.dqn.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm
