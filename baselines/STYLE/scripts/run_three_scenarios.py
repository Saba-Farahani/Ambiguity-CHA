#!/usr/bin/env python3
"""
Run the three STYLE transfer scenarios on a target domain (mental health / food).

Scenario 1 — Zero-shot: train on source domains only, evaluate on target.
Scenario 2 — Fine-tuned: load Scenario 1 checkpoint, fine-tune on target train split.
Scenario 3 — In-domain: train from scratch on target train split only.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from style.config import Config
from style.training.mdt import MDTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Run STYLE three-scenario experiments")
    parser.add_argument(
        "--target-train",
        required=True,
        help="JSON file with target-domain training cases",
    )
    parser.add_argument(
        "--target-test",
        required=True,
        help="JSON file with target-domain test cases",
    )
    parser.add_argument(
        "--source-domains",
        nargs="+",
        default=["clariq", "opendialkg"],
    )
    parser.add_argument("--episodes", type=int, default=1800)
    parser.add_argument("--finetune-episodes", type=int, default=600)
    parser.add_argument("--output-dir", default="saved_models/scenarios")
    return parser.parse_args()


def load_json_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    return data


def train_and_save(config, train_data, val_data, test_data, tag, episodes):
    config.NUM_EPOCHS = episodes
    save_dir = os.path.join(config.MODEL_SAVE_DIR, tag)
    config.MODEL_SAVE_DIR = save_dir
    os.makedirs(save_dir, exist_ok=True)

    trainer = MDTTrainer(config)
    trainer.train_data = train_data
    trainer.val_data = val_data
    trainer.test_data = test_data
    trainer.train(episodes)
    ckpt = os.path.join(save_dir, f"{tag}.pt")
    trainer.disp_model.save(ckpt)
    metrics = trainer.evaluate() if hasattr(trainer, "evaluate") else {}
    return ckpt, metrics


def main():
    args = parse_args()
    config = Config()
    config.MODEL_SAVE_DIR = args.output_dir
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    target_train = load_json_cases(args.target_train)
    target_test = load_json_cases(args.target_test)
    val_split = max(1, len(target_train) // 10)
    target_val = target_train[:val_split]
    target_train = target_train[val_split:]

    results = {}

    print("=" * 60)
    print("Scenario 1: Zero-shot (source domains → target test)")
    config_s1 = Config()
    config_s1.DOMAINS = [d.lower() for d in args.source_domains]
    config_s1.MODEL_SAVE_DIR = args.output_dir
    trainer_s1 = MDTTrainer(config_s1)
    trainer_s1.train(args.episodes)
    ckpt_s1 = os.path.join(args.output_dir, "scenario1_zero_shot.pt")
    trainer_s1.disp_model.save(ckpt_s1)
    trainer_s1.test_data = target_test
    results["scenario_1"] = trainer_s1.evaluate() if hasattr(trainer_s1, "evaluate") else {}

    print("=" * 60)
    print("Scenario 2: Fine-tuned transfer")
    config_s2 = Config()
    config_s2.MODEL_SAVE_DIR = args.output_dir
    config_s2.EPSILON = 0.2
    trainer_s2 = MDTTrainer(config_s2)
    trainer_s2.disp_model.load(ckpt_s1)
    trainer_s2.train_data = target_train
    trainer_s2.val_data = target_val
    trainer_s2.test_data = target_test
    trainer_s2.train(args.finetune_episodes)
    ckpt_s2 = os.path.join(args.output_dir, "scenario2_finetuned.pt")
    trainer_s2.disp_model.save(ckpt_s2)
    results["scenario_2"] = trainer_s2.evaluate() if hasattr(trainer_s2, "evaluate") else {}

    print("=" * 60)
    print("Scenario 3: In-domain from scratch")
    config_s3 = Config()
    config_s3.MODEL_SAVE_DIR = args.output_dir
    trainer_s3 = MDTTrainer(config_s3)
    trainer_s3.train_data = target_train
    trainer_s3.val_data = target_val
    trainer_s3.test_data = target_test
    trainer_s3.train(args.episodes)
    ckpt_s3 = os.path.join(args.output_dir, "scenario3_in_domain.pt")
    trainer_s3.disp_model.save(ckpt_s3)
    results["scenario_3"] = trainer_s3.evaluate() if hasattr(trainer_s3, "evaluate") else {}

    out_path = os.path.join(args.output_dir, "scenario_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
