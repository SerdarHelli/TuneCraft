#!/usr/bin/env python3
"""
Complete GRPO training pipeline for MedGemma VLM
"""

import os
import sys

def main():
    print("=== MedGemma GRPO Training Pipeline ===")
    print("Using HuggingFace Iterable Datasets for 600k+ images")
    
    # Step 1: Test dataset creation
    print("\n1. Testing dataset creation...")
    try:
        from create_datasets import main as create_datasets_main
        ds_train, ds_val, ds_test = create_datasets_main()
        print("✅ Iterable datasets created successfully")
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
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
    print("💾 Memory efficient: Images loaded on-demand")
    print("🚀 Scalable: Handles any dataset size")
    return 0

if __name__ == "__main__":
    sys.exit(main())