#!/usr/bin/env python3
"""Evaluate trained STYLE checkpoints on mental-health and food ambiguity datasets."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluate_new_datasets import main

if __name__ == "__main__":
    main()
