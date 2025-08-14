#!/usr/bin/env python3
"""
Test script for PyTorch DataLoader implementation
"""

import sys
import os
from torch_dataloader import test_dataloader
from config import *

def main():
    print("🧪 Testing PyTorch DataLoader Implementation")
    print("=" * 60)
    
    # Check if data files exist
    print("📁 Checking data files...")
    for name, path in [("Train", TRAIN_JSON), ("Val", VAL_JSON), ("Test", TEST_JSON)]:
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} (NOT FOUND)")
            print(f"     Please update the path in config.py")
    
    print(f"\n🖼️  Image configuration:")
    print(f"  Target size: {IMAGE_CONFIG['target_size']}")
    print(f"  Use only first image: {USE_ONLY_FIRST_IMAGE}")
    
    print(f"\n⚙️  DataLoader configuration:")
    print(f"  Batch size: {DATALOADER_CONFIG['batch_size']}")
    print(f"  Num workers: {DATALOADER_CONFIG['num_workers']}")
    print(f"  Pin memory: {DATALOADER_CONFIG['pin_memory']}")
    
    try:
        # Test the dataloader
        print(f"\n🚀 Testing DataLoader...")
        train_loader, val_loader, test_loader = test_dataloader()
        
        print(f"\n📊 DataLoader Statistics:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Test multiple batches
        print(f"\n🔄 Testing multiple batches...")
        for i, batch in enumerate(train_loader):
            if i >= 3:  # Test first 3 batches
                break
            print(f"  Batch {i+1}: {len(batch['prompt'])} samples, images: {[img.size for img in batch['image']]}")
        
        print(f"\n🎉 All tests passed!")
        print(f"💡 Ready for training with PyTorch DataLoader")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print(f"Please check your data paths and configuration.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)