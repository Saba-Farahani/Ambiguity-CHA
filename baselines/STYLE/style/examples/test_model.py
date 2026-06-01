import os
import json
import torch
import numpy as np
from style.training.mdt import MDTTrainer
from style.config import Config


def compute_metrics(predictions, targets):
    """Compute basic evaluation metrics."""
    metrics = {
        "accuracy": np.mean([p == t for p, t in zip(predictions, targets)]),
        "precision": np.mean([p == t for p, t in zip(predictions, targets) if p == 1]),
        "recall": np.mean([p == t for p, t in zip(predictions, targets) if t == 1]),
        "f1": 2
        * np.mean([p == t for p, t in zip(predictions, targets)])
        / (1 + np.mean([p == t for p, t in zip(predictions, targets)])),
    }
    return metrics


def main():
    # Initialize configuration
    config = Config()

    # Initialize trainer
    trainer = MDTTrainer(config)

    # Load the best model
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pt")
    if os.path.exists(model_path):
        trainer.disp_model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        return

    # Test files
    test_files = {
        "clariq": "/home/ailaty3088@id.sdsu.edu/STYLE/data/clariq_test.json",
        "opendialkg": "/home/ailaty3088@id.sdsu.edu/STYLE/data/opendialkg_test.json",
    }

    # Run evaluation on each domain
    domain_metrics = {}
    for domain, test_file in test_files.items():
        print(f"\nEvaluating on {domain} domain...")

        try:
            # Load test data
            with open(test_file, "r") as f:
                test_data = json.load(f)

            print(f"Loaded {len(test_data)} test examples from {test_file}")

            # Run evaluation
            predictions = []
            targets = []

            for example in test_data:
                # Get model prediction
                state = trainer._prepare_state(example)
                action, _ = trainer.disp_model.select_action(state)
                predictions.append(action.item())

                # Get target
                target = example.get("target_action", 0)
                targets.append(target)

            # Compute metrics
            metrics = compute_metrics(predictions, targets)
            domain_metrics[domain] = metrics

            # Print metrics
            print(f"\n{domain} Domain Metrics:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

        except Exception as e:
            print(f"Error evaluating {domain}: {str(e)}")
            continue

    # Compute overall metrics
    overall_metrics = {
        "accuracy": np.mean([m["accuracy"] for m in domain_metrics.values()]),
        "precision": np.mean([m["precision"] for m in domain_metrics.values()]),
        "recall": np.mean([m["recall"] for m in domain_metrics.values()]),
        "f1": np.mean([m["f1"] for m in domain_metrics.values()]),
    }

    # Print overall metrics
    print("\nOverall Metrics:")
    for metric_name, value in overall_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    # Save metrics
    metrics_path = os.path.join(config.MODEL_SAVE_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {"domain_metrics": domain_metrics, "overall_metrics": overall_metrics},
            f,
            indent=4,
        )

    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
