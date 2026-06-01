"""
Configuration optimization for STYLE training stability.
Implements suggestions from training analysis to improve model performance.
"""

import os
import json
from style.config import Config


def create_optimized_config(analysis_results: dict = None) -> Config:
    """Create an optimized configuration based on training analysis.

    Args:
        analysis_results: Optional analysis results to guide optimization

    Returns:
        Optimized configuration object
    """
    config = Config()

    # Apply stability improvements based on analysis
    print("🔧 Applying training stability optimizations...")

    # 1. Reduce learning rate for better stability (from 1e-3 to 3e-4)
    config.LEARNING_RATE = 3e-4
    print(f"  • Learning rate: {config.LEARNING_RATE}")

    # 2. Adjust epsilon decay for better exploration
    config.EPSILON_START = 1.0
    config.EPSILON_END = 0.1
    config.EPSILON_DECAY = 0.999  # Slower decay for more exploration
    config.EPSILON_MIN = 0.05
    print(f"  • Epsilon decay: {config.EPSILON_DECAY}")

    # 3. More frequent target network updates for stability
    config.TARGET_UPDATE = 5  # Update every 5 steps instead of 10
    print(f"  • Target update frequency: {config.TARGET_UPDATE}")

    # 4. Adjust reward structure for better action balance
    config.REWARD_ASK = 0.05  # Reduced from 0.1
    config.REWARD_CLARIFY = 0.1  # Reduced from 0.2
    config.REWARD_ANSWER = 1.0  # Keep high reward for successful answers
    config.REWARD_SUCCESS = 2.0  # Keep high reward for success
    config.FAILURE_REWARD = -0.05  # Reduced penalty
    config.STEP_PENALTY = -0.005  # Reduced step penalty
    print(f"  • Reward structure optimized")

    # 5. Increase buffer size for more stable training
    config.MEMORY_SIZE = 20000  # Increased from 10000
    config.DQN_BUFFER_SIZE = 20000
    print(f"  • Memory size: {config.MEMORY_SIZE}")

    # 6. Adjust batch size for stability
    config.BATCH_SIZE = 16  # Reduced from 32 for more stable gradients
    print(f"  • Batch size: {config.BATCH_SIZE}")

    # 7. Enable gradient clipping (already implemented in DISP)
    print(f"  • Gradient clipping: enabled (max_norm=1.0)")

    # 8. Add learning rate scheduling
    config.USE_LR_SCHEDULER = True
    config.LR_SCHEDULER_PATIENCE = 5
    config.LR_SCHEDULER_FACTOR = 0.5
    print(f"  • Learning rate scheduler: enabled")

    # 9. Enhanced monitoring
    config.MONITORING_INTERVAL = 50  # Monitor every 50 steps
    config.STABILITY_CHECK_INTERVAL = 100
    print(f"  • Enhanced monitoring: enabled")

    # 10. Debug mode for detailed logging
    config.DEBUG = True
    print(f"  • Debug mode: enabled")

    return config


def create_quick_optimized_config() -> Config:
    """Create optimized configuration for quick training."""
    config = create_optimized_config()

    # Quick training specific optimizations
    config.NUM_EPOCHS = 5
    config.BATCH_SIZE = 8
    config.MEMORY_SIZE = 5000
    config.DQN_BUFFER_SIZE = 5000

    # Faster convergence for quick training
    config.LEARNING_RATE = 5e-4
    config.EPSILON_DECAY = 0.995

    print(f"  • Quick training mode: {config.NUM_EPOCHS} epochs")

    return config


def save_optimized_config(config: Config, filename: str = "optimized_config.json"):
    """Save optimized configuration to file.

    Args:
        config: Configuration object to save
        filename: Output filename
    """
    config_dict = {}

    # Extract relevant configuration parameters
    relevant_params = [
        "LEARNING_RATE",
        "EPSILON_START",
        "EPSILON_END",
        "EPSILON_DECAY",
        "EPSILON_MIN",
        "TARGET_UPDATE",
        "REWARD_ASK",
        "REWARD_CLARIFY",
        "REWARD_ANSWER",
        "REWARD_SUCCESS",
        "FAILURE_REWARD",
        "STEP_PENALTY",
        "MEMORY_SIZE",
        "DQN_BUFFER_SIZE",
        "BATCH_SIZE",
        "USE_LR_SCHEDULER",
        "LR_SCHEDULER_PATIENCE",
        "LR_SCHEDULER_FACTOR",
        "MONITORING_INTERVAL",
        "STABILITY_CHECK_INTERVAL",
        "DEBUG",
        "NUM_EPOCHS",
    ]

    for param in relevant_params:
        if hasattr(config, param):
            config_dict[param] = getattr(config, param)

    # Save to file
    with open(filename, "w") as f:
        json.dump(config_dict, f, indent=4)

    print(f"✅ Optimized configuration saved to {filename}")


def load_optimized_config(filename: str = "optimized_config.json") -> Config:
    """Load optimized configuration from file.

    Args:
        filename: Configuration file to load

    Returns:
        Configuration object
    """
    if not os.path.exists(filename):
        print(
            f"⚠️ Configuration file {filename} not found. Creating new optimized config."
        )
        return create_optimized_config()

    with open(filename, "r") as f:
        config_dict = json.load(f)

    config = Config()

    # Apply loaded parameters
    for param, value in config_dict.items():
        if hasattr(config, param):
            setattr(config, param, value)

    print(f"✅ Loaded optimized configuration from {filename}")
    return config


def compare_configs(original_config: Config, optimized_config: Config):
    """Compare original and optimized configurations.

    Args:
        original_config: Original configuration
        optimized_config: Optimized configuration
    """
    print("\n📊 Configuration Comparison")
    print("=" * 50)

    comparison_params = [
        ("Learning Rate", "LEARNING_RATE"),
        ("Epsilon Decay", "EPSILON_DECAY"),
        ("Target Update", "TARGET_UPDATE"),
        ("Batch Size", "BATCH_SIZE"),
        ("Memory Size", "MEMORY_SIZE"),
        ("Reward Ask", "REWARD_ASK"),
        ("Reward Clarify", "REWARD_CLARIFY"),
        ("Reward Answer", "REWARD_ANSWER"),
        ("Failure Reward", "FAILURE_REWARD"),
        ("Step Penalty", "STEP_PENALTY"),
    ]

    for param_name, param_key in comparison_params:
        if hasattr(original_config, param_key) and hasattr(optimized_config, param_key):
            original_val = getattr(original_config, param_key)
            optimized_val = getattr(optimized_config, param_key)

            change = "→"
            if original_val != optimized_val:
                change = "↗️" if optimized_val > original_val else "↘️"

            print(f"{param_name:15} {original_val:8.4f} {change} {optimized_val:8.4f}")


def main():
    """Main function to demonstrate configuration optimization."""
    print("🚀 STYLE Configuration Optimization")
    print("=" * 50)

    # Create original config for comparison
    original_config = Config()

    # Create optimized config
    optimized_config = create_optimized_config()

    # Compare configurations
    compare_configs(original_config, optimized_config)

    # Save optimized configuration
    save_optimized_config(optimized_config)

    # Create quick training config
    quick_config = create_quick_optimized_config()
    save_optimized_config(quick_config, "quick_optimized_config.json")

    print("\n✅ Configuration optimization completed!")
    print("\n📁 Generated files:")
    print("  • optimized_config.json - Full optimized configuration")
    print("  • quick_optimized_config.json - Quick training configuration")

    print("\n💡 Usage:")
    print("  • Use optimized_config.json for full training")
    print("  • Use quick_optimized_config.json for quick testing")
    print("  • These configs implement all stability improvements from the analysis")


if __name__ == "__main__":
    main()
