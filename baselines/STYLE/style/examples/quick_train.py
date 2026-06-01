import os
import torch
import json
import time
from collections import deque
from style.training.mdt import MDTTrainer, format_time
from style.config import Config
from style.examples.optimize_config import create_quick_optimized_config
import argparse
import pandas as pd


def create_quick_config():
    """Create a configuration for quick training with stability optimizations."""
    print("🔧 Loading optimized configuration for quick training...")

    # Use optimized configuration
    config = create_quick_optimized_config()

    # Create necessary directories
    config.CHECKPOINT_DIR = "checkpoints/quick_train"
    config.LOG_DIR = "logs/quick_train"
    config.MODEL_SAVE_DIR = "saved_models/quick_train"

    # Disable tensorboard if not available
    try:
        from torch.utils.tensorboard import SummaryWriter

        config.USE_TENSORBOARD = True
    except ImportError:
        print("TensorBoard not available. Disabling visualization.")
        config.USE_TENSORBOARD = False

    return config


def load_quick_data():
    """Load a small subset of data for quick training."""
    data_dir = os.path.join("data")

    # Try to load domain-specific data first
    train_data = []
    val_data = []
    test_data = []

    # Load ClariQ data
    clariq_train = load_data_file(os.path.join(data_dir, "clariq_train.json"))
    clariq_dev = load_data_file(os.path.join(data_dir, "clariq_dev.json"))
    clariq_test = load_data_file(os.path.join(data_dir, "clariq_test.json"))

    # Load OpenDialKG data
    opendialkg_train = load_data_file(os.path.join(data_dir, "opendialkg_train.json"))
    opendialkg_dev = load_data_file(os.path.join(data_dir, "opendialkg_dev.json"))
    opendialkg_test = load_data_file(os.path.join(data_dir, "opendialkg_test.json"))

    # Combine data
    train_data.extend(clariq_train)
    train_data.extend(opendialkg_train)
    val_data.extend(clariq_dev)
    val_data.extend(opendialkg_dev)
    test_data.extend(clariq_test)
    test_data.extend(opendialkg_test)

    # Take small subset (20% of data for quick training)
    train_size = max(1, len(train_data) // 5)
    val_size = max(1, len(val_data) // 5)
    test_size = max(1, len(test_data) // 5)

    return {
        "train": train_data[:train_size],
        "val": val_data[:val_size],
        "test": test_data[:test_size],
    }


def load_data_file(file_path: str) -> list:
    """Load data from a JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} examples from {file_path}")
        return data
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return []


def load_csv_data(file_path):
    """Load data from a CSV file and convert to list of dicts."""
    try:
        df = pd.read_csv(file_path)
        data = df.to_dict(orient='records')
        print(f"✅ Loaded {len(data)} examples from {file_path}")
        return data
    except Exception as e:
        print(f"⚠️ Error loading {file_path}: {e}")
        return []


def print_training_summary(config, data):
    """Print a summary of the training configuration and data."""
    print("\n" + "=" * 60)
    print("🚀 QUICK TRAINING CONFIGURATION")
    print("=" * 60)

    print(f"\n📋 Training Parameters:")
    print(f"  • Epochs: {config.NUM_EPOCHS}")
    print(f"  • Batch Size: {config.BATCH_SIZE}")
    print(f"  • Learning Rate: {config.LEARNING_RATE}")
    print(f"  • Epsilon: {config.EPSILON}")
    print(f"  • Hidden Dim: {config.HIDDEN_DIM}")
    print(f"  • Memory Size: {config.MEMORY_SIZE}")

    print(f"\n📊 Dataset Information:")
    print(f"  • Train samples: {len(data['train'])}")
    print(f"  • Validation samples: {len(data['val'])}")
    print(f"  • Test samples: {len(data['test'])}")
    print(f"  • Total steps per epoch: {len(data['train'])}")
    print(f"  • Total training steps: {len(data['train']) * config.NUM_EPOCHS}")

    print(f"\n📁 Output Directories:")
    print(f"  • Checkpoints: {config.CHECKPOINT_DIR}")
    print(f"  • Logs: {config.LOG_DIR}")
    print(f"  • Models: {config.MODEL_SAVE_DIR}")

    print(f"\n🔧 Features Enabled:")
    print(f"  • ETA Tracking: ✅")
    print(f"  • Checkpointing: ✅")
    print(f"  • Strategy Diversity: ✅")
    print(f"  • Debug Mode: {'✅' if config.DEBUG else '❌'}")
    print(f"  • WandB Logging: {'✅' if hasattr(config, 'WANDB_PROJECT') else '❌'}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, help='Path to training CSV')
    parser.add_argument('--val', type=str, help='Path to validation CSV')
    parser.add_argument('--test', type=str, help='Path to test CSV')
    args = parser.parse_args()

    print("🚀 Starting Enhanced Quick Training with ETA Tracking")
    print("=" * 60)

    # Create quick config
    config = create_quick_config()

    # Create necessary directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    # Load data
    if args.train and args.val and args.test:
        print(f"\n📂 Loading diagnosis data from CSVs...")
        train_data = load_csv_data(args.train)
        val_data = load_csv_data(args.val)
        test_data = load_csv_data(args.test)
        data = {"train": train_data, "val": val_data, "test": test_data}
    else:
        print("\n📂 Loading training data (default quick data)...")
        data = load_quick_data()

    # Print training summary
    print_training_summary(config, data)

    # Initialize trainer
    print(f"\n🔧 Initializing MDT Trainer...")
    trainer = MDTTrainer(config)

    # If using diagnosis CSVs, set the data on the trainer
    if args.train and args.val and args.test:
        trainer.train_data = data['train']
        trainer.val_data = data['val']
        trainer.test_data = data['test']

    # Check for existing checkpoints
    checkpoint_files = [
        f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith("checkpoint_")
    ]
    if checkpoint_files:
        print(f"\n📁 Found {len(checkpoint_files)} existing checkpoint(s):")
        for checkpoint in sorted(checkpoint_files):
            print(f"  - {checkpoint}")
        print(f"\n🔄 Quick training mode: Starting fresh (ignoring existing checkpoints)")

    # Start training with ETA tracking
    print(f"\n🎯 Starting training for {config.NUM_EPOCHS} epochs...")
    print("📊 Progress will be displayed with ETA and loss information")
    print("💾 Checkpoints will be saved automatically")
    print("-" * 60)

    start_time = time.time()
    trainer.train(num_epochs=config.NUM_EPOCHS)
    total_time = time.time() - start_time

    # Print training completion summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)

    print(f"\n⏱️ Total Training Time: {format_time(total_time)}")
    print(f"📊 Average Time per Epoch: {format_time(total_time / config.NUM_EPOCHS)}")

    # Evaluate
    print(f"\n🧪 Running final evaluation...")
    metrics = trainer.evaluate()

    # Print final results
    print(f"\n🏁 Final Evaluation Results:")
    print("-" * 40)
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    # Save results
    results_dir = os.path.join(config.MODEL_SAVE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(results_dir, "quick_train_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": {
                    k: v for k, v in config.__dict__.items() if not k.startswith("_")
                },
                "training_time": total_time,
                "epochs": config.NUM_EPOCHS,
                "data_sizes": {k: len(v) for k, v in data.items()},
            },
            f,
            indent=4,
        )

    print(f"\n📁 Results saved to: {results_dir}")
    print(f"📊 Training log: {trainer.log_file_path}")
    print(f"💾 Models: {config.MODEL_SAVE_DIR}")
    print(f"📁 Checkpoints: {config.CHECKPOINT_DIR}")

    print("\n🎉 Quick training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
