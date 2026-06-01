#!/usr/bin/env python3
"""Fine-tune a pre-trained STYLE checkpoint on the diagnosis dataset."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from style.examples.diagnosis_finetune import main

if __name__ == "__main__":
    main()
