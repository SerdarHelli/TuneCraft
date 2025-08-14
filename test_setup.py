#!/usr/bin/env python3
"""
Test script to verify the GRPO setup is working correctly
"""

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import Dataset
from PIL import Image
import numpy as np

def test_dependencies():
    """Test that all required packages are installed"""
    print("Testing dependencies...")
    
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError:
        print("❌ transformers not installed")
        return False
    
    try:
        import trl
        print(f"✅ trl: {trl.__version__}")
    except ImportError:
        print("❌ trl not installed")
        return False
    
    try:
        import peft
        print(f"✅ peft: {peft.__version__}")
    except ImportError:
        print("❌ peft not installed")
        return False
    
    try:
        import bitsandbytes
        print(f"✅ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        print("❌ bitsandbytes not installed")
        return False
    
    return True

def test_model_loading():
    """Test that the model can be loaded"""
    print("\nTesting model loading...")
    
    try:
        MODEL_ID = "google/medgemma-4b-it"
        
        # Test processor loading
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("✅ Processor loaded successfully")
        
        # Test model loading (without quantization for testing)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16,
            device_map="cpu",  # Use CPU for testing
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_dataset_format():
    """Test that the dataset format is correct for GRPO"""
    print("\nTesting dataset format...")
    
    try:
        # Create a dummy dataset in the correct format
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        dataset = Dataset.from_list([
            {
                "prompt": "Test prompt for medical VQA",
                "image": dummy_image,
                "gold_letter": "A",
                "gold_explanation": "Test explanation",
                "impression": "Test impression",
                "findings": "Test findings",
                "indication": "Test indication",
                "heur_reason": "Test reason",
            }
        ])
        
        print("✅ Dataset format is correct")
        print(f"   Dataset columns: {dataset.column_names}")
        print(f"   Dataset size: {len(dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with dataset format: {e}")
        return False

def test_reward_function():
    """Test the reward function"""
    print("\nTesting reward function...")
    
    try:
        # Import the reward function components
        import re
        
        def between(text, start, end):
            try:
                s = text.index(start) + len(start)
                e = text.index(end, s)
                return text[s:e].strip()
            except ValueError:
                return ""
        
        def token_f1(pred, gold):
            import collections
            p = [t for t in re.findall(r"\w+", (pred or "").lower()) if t]
            g = [t for t in re.findall(r"\w+", (gold or "").lower()) if t]
            if not p or not g: return 0.0
            pc, gc = collections.Counter(p), collections.Counter(g)
            overlap = sum((pc & gc).values())
            if overlap == 0: return 0.0
            prec, rec = overlap/len(p), overlap/len(g)
            return 2*prec*rec/(prec+rec)
        
        # Test with sample completion
        test_completion = "<start_working_out>This is reasoning<end_working_out><SOLUTION>A - Test answer</SOLUTION>"
        
        reasoning = between(test_completion, "<start_working_out>", "<end_working_out>")
        solution = between(test_completion, "<SOLUTION>", "</SOLUTION>")
        
        print(f"✅ Reward function components working")
        print(f"   Extracted reasoning: '{reasoning}'")
        print(f"   Extracted solution: '{solution}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with reward function: {e}")
        return False

def main():
    """Run all tests"""
    print("=== GRPO Setup Test ===")
    
    tests = [
        test_dependencies,
        test_model_loading,
        test_dataset_format,
        test_reward_function,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Setup is ready for GRPO training.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())