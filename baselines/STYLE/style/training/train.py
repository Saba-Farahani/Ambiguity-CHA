"""
Main training script.
"""

import argparse
import os
import wandb
import torch
from typing import List
from ..training.mdt import MDTTrainer
from ..config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train the STYLE model")
    parser.add_argument(
        "--episodes", type=int, default=1800, help="Number of training episodes"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["ClariQ", "OpenDialKG"],
        help="List of domains to train on (ClariQ and/or OpenDialKG)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=Config.LEARNING_RATE,
        help="Learning rate for optimization",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=Config.MEMORY_SIZE,
        help="Size of replay memory",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use (0-3)")
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()

    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("CUDA is not available. Using CPU.")

    # Print current working directory and environment variables
    print(f"Current working directory: {os.getcwd()}")
    print(f"Environment variables: {dict(os.environ)}")

    # Initialize configuration
    config = Config()
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.MEMORY_SIZE = args.memory_size
    config.NUM_EPOCHS = args.episodes

    # Initialize dataset manager (optional — MDTTrainer loads data internally)
    from ..data.dataset_manager_full import DatasetManager
    dataset_manager = DatasetManager()

    # Initialize trainer with all required arguments
    trainer = MDTTrainer(config)

    # Load documents for each domain
    print("Loading documents for each domain...")
    for domain in args.domains:
        print(f"Loading documents for domain: {domain}")
        trainer.load_documents_for_domain(domain.lower())

    # Train the model
    trainer.train(args.episodes)

    # Save the trained model
    trainer.save_model("trained_model.pth")


if __name__ == "__main__":
    main()
