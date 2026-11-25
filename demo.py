"""
demo.py - Interactive Demonstration Runner for Direct Volume Rendering Engine.
Course: IS F311 Computer Graphics
Authors: Tarun P Krishnamurthy, Harsh Gunda, Ojas Dubey

Usage:
    python demo.py                     # Launch Modern Interactive Web Dashboard
    python demo.py --cli --dataset brain_mri --method levoy --output output/brain.png
    python demo.py --test              # Run complete mathematical unit test suite
"""

import os
import sys

# Ensure root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    main()
