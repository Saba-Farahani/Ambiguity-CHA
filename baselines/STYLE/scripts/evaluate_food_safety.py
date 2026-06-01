#!/usr/bin/env python3
"""Evaluate STYLE on the food-safety benchmark."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate_food_safety import evaluate

if __name__ == "__main__":
    evaluate()
