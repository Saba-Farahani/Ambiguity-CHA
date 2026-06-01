"""
Example script demonstrating how to use the DatasetManager for setting up dataset splits.
"""

import os
import json
from ..data.dataset_manager_full import DatasetManager
from ..config import Config


def create_sample_data(data_dir: str, num_samples: int = 1000):
    """Create sample data for demonstration purposes."""
    os.makedirs(data_dir, exist_ok=True)

    # Create sample data for two domains
    domains = ["domain1", "domain2"]

    for domain in domains:
        data = []
        for i in range(num_samples):
            sample = {
                "query_history": [
                    f"query_{i}_{j}" for j in range(3)
                ],  # 3 queries per sample
                "documents": [
                    f"doc_{i}_{j}" for j in range(Config.TOP_K_DOCS)
                ],  # TOP_K_DOCS documents
                "retrieval_scores": [
                    0.8 - j * 0.1 for j in range(Config.TOP_K_DOCS)
                ],  # Decreasing scores
                "target_action": 1 if i % 2 == 0 else 0,  # Alternating actions
                "success": i % 3 == 0,  # 1/3 of samples are successful
                "num_turns": i % Config.MAX_TURNS,  # Varying number of turns
            }
            data.append(sample)

        # Save domain data
        file_path = os.path.join(data_dir, f"{domain}_data.json")
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Created {num_samples} samples for {domain}")


def main():
    # Create sample data
    data_dir = os.path.join(Config.DATA_DIR, "sample")
    create_sample_data(data_dir)

    # Initialize dataset manager
    dataset_manager = DatasetManager(data_dir=data_dir)

    # Load domain data
    domain_files = {
        "domain1": os.path.join(data_dir, "domain1_data.json"),
        "domain2": os.path.join(data_dir, "domain2_data.json"),
    }
    dataset_manager.load_domains(domain_files)

    # Create splits
    dataset_manager.create_splits(
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )

    # Save splits to disk
    dataset_manager.save_splits()

    # Print dataset statistics
    stats = dataset_manager.get_domain_stats()
    print("\nDataset Statistics:")
    for domain_name, domain_stats in stats.items():
        print(f"\nDomain: {domain_name}")
        print(f"Train samples: {domain_stats['train']}")
        print(f"Val samples: {domain_stats['val']}")
        print(f"Test samples: {domain_stats['test']}")

    # Demonstrate getting a batch
    batch = dataset_manager.get_domain_batch(
        domain_name="domain1", split="train", batch_size=Config.BATCH_SIZE
    )
    print("\nSample batch structure:")
    for key, value in batch.items():
        if isinstance(value, list):
            print(f"{key}: list of length {len(value)}")
        else:
            print(f"{key}: tensor of shape {value.shape}")


if __name__ == "__main__":
    main()
