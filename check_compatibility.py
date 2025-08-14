"""
Check compatibility with PyTorch 2.5.1+cu124 and other dependencies
"""
import sys

def check_pytorch():
    """Check PyTorch installation and CUDA support"""
    print("🔍 Checking PyTorch...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("⚠️  CUDA not available - training will be very slow")
        
        return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_transformers():
    """Check Transformers library"""
    print("\n🔍 Checking Transformers...")
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        return True
    except ImportError:
        print("❌ Transformers not installed")
        return False

def check_unsloth():
    """Check Unsloth installation"""
    print("\n🔍 Checking Unsloth...")
    try:
        import unsloth
        print("✅ Unsloth installed")
        
        # Test FastVisionModel import
        from unsloth import FastVisionModel
        print("✅ FastVisionModel available")
        
        return True
    except ImportError as e:
        print(f"❌ Unsloth not installed: {e}")
        return False

def check_other_deps():
    """Check other dependencies"""
    print("\n🔍 Checking other dependencies...")
    
    deps = [
        ("datasets", "datasets"),
        ("accelerate", "accelerate"), 
        ("trl", "trl"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("PIL", "Pillow")
    ]
    
    all_good = True
    for module_name, package_name in deps:
        try:
            if module_name == "PIL":
                import PIL
                print(f"✅ {package_name}: {PIL.__version__}")
            else:
                module = __import__(module_name)
                version = getattr(module, '__version__', 'unknown')
                print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name} not installed")
            all_good = False
    
    return all_good

def check_memory_requirements():
    """Check if system meets memory requirements"""
    print("\n🔍 Checking memory requirements...")
    
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                gpu_name = torch.cuda.get_device_name(i)
                
                if gpu_memory >= 24:
                    print(f"✅ {gpu_name}: {gpu_memory:.1f} GB - Excellent for training")
                elif gpu_memory >= 16:
                    print(f"✅ {gpu_name}: {gpu_memory:.1f} GB - Good for training")
                elif gpu_memory >= 12:
                    print(f"⚠️  {gpu_name}: {gpu_memory:.1f} GB - May need smaller batch size")
                else:
                    print(f"❌ {gpu_name}: {gpu_memory:.1f} GB - Insufficient for training")
        else:
            print("❌ No CUDA GPUs available")
    except:
        print("❌ Could not check GPU memory")

def main():
    """Run all compatibility checks"""
    print("🏥 MedGemma SFT Compatibility Check")
    print("=" * 50)
    
    checks = [
        check_pytorch(),
        check_transformers(),
        check_unsloth(),
        check_other_deps()
    ]
    
    check_memory_requirements()
    
    print("\n" + "=" * 50)
    if all(checks):
        print("🎉 All compatibility checks passed!")
        print("✅ Ready for MedGemma SFT training")
        print("\nNext steps:")
        print("  python test_dataset.py    # Test datasets")
        print("  python train.py           # Start training")
    else:
        print("❌ Some compatibility issues found")
        print("Please install missing dependencies:")
        print("  python install_unsloth.py")

if __name__ == "__main__":
    main()