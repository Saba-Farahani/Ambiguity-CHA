#!/usr/bin/env python3
"""Train STYLE DISP from scratch on the Synthea diagnosis dataset."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from train_diagnosis_style_model import main

if __name__ == "__main__":
    main()
