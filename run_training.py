#!/usr/bin/env python3
"""
Complete GRPO training pipeline for MedGemma VLM
"""

import os
import sys

def main():
    print("=== MedGemma GRPO Training Pipeline ===")
    
    # Step 1: Prepare datasets
    print("\n1. Preparing datasets...")
    try:
        exec(open("dataset_pre.py").read())
        print("✅ Datasets prepared successfully")
    except Exception as e:
        print(f"❌ Error preparing datasets: {e}")
        return 1
    
    # Step 2: Run GRPO training
    print("\n2. Starting GRPO training...")
    try:
        exec(open("train.py").read())
        print("✅ Training completed successfully")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return 1
    
    print("\n🎉 Training pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())