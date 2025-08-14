"""
Install Unsloth and dependencies for SFT training
"""
import subprocess
import sys

def run_command(command):
    """Run a command and print output"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Install Unsloth and dependencies"""
    
    print("Installing Unsloth for efficient fine-tuning...")
    
    # Check existing PyTorch version
    print("\n1. Checking PyTorch installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} already installed")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
    except ImportError:
        print("Installing PyTorch with CUDA support...")
        torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
        run_command(torch_cmd)
    
    # Install Unsloth
    print("\n2. Installing Unsloth...")
    unsloth_cmd = "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
    run_command(unsloth_cmd)
    
    # Install other dependencies
    print("\n3. Installing other dependencies...")
    deps = [
        "transformers>=4.46.0",
        "datasets>=2.14.0", 
        "accelerate>=0.24.0",
        "trl>=0.12.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "Pillow>=9.0.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0"
    ]
    
    for dep in deps:
        run_command(f"pip install {dep}")
    
    print("\n4. Verifying installation...")
    
    # Test imports
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} installed")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        import unsloth
        print("✅ Unsloth installed successfully")
        
        from transformers import AutoTokenizer
        print("✅ Transformers installed")
        
        from trl import SFTTrainer
        print("✅ TRL installed")
        
        from peft import LoraConfig
        print("✅ PEFT installed")
        
        print("\n🎉 All dependencies installed successfully!")
        print("\nYou can now run:")
        print("  python train.py")
        print("  python test_dataset.py  # Test iterable datasets")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some dependencies may not have installed correctly.")

if __name__ == "__main__":
    main()