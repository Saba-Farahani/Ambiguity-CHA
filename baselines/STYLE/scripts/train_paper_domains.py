#!/usr/bin/env python3
"""Train DISP with MDT on ClariQ and/or OpenDialKG (paper source domains)."""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from style.training.mdt import MDTTrainer
from style.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Train STYLE DISP on paper domains")
    parser.add_argument("--episodes", type=int, default=1800)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["clariq", "opendialkg"],
        help="Training domains",
    )
    parser.add_argument("--output-dir", default="saved_models/paper_domains")
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    config.NUM_EPOCHS = args.episodes
    config.DOMAINS = [d.lower() for d in args.domains]
    config.MODEL_SAVE_DIR = args.output_dir
    config.DEBUG = False

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    print(f"Training on {config.DOMAINS} for {config.NUM_EPOCHS} episodes")
    print(f"State dim: {config.STATE_INPUT_DIM}, LR: {config.LEARNING_RATE}")

    trainer = MDTTrainer(config)
    trainer.train(config.NUM_EPOCHS)
    trainer._save_model("final_model.pt")
    print(f"Model saved to {config.MODEL_SAVE_DIR}/final_model.pt")


if __name__ == "__main__":
    main()
