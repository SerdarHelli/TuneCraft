"""
Complete setup and training script for MedGemma SFT
"""
import os
import sys
import subprocess

def run_command(command, description):
    """Run a command with description"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        return False

def main():
    """Complete setup and training pipeline"""
    
    print("🏥 MedGemma SFT Training Pipeline")
    print("Using Unsloth for efficient fine-tuning with iterable datasets")
    print("=" * 60)
    
    # Step 1: Install dependencies
    if not run_command("python install_unsloth.py", "Installing Unsloth and dependencies"):
        print("❌ Failed to install dependencies. Please check the error messages above.")
        return
    
    # Step 2: Download data
    if not run_command("python download_data.py", "Downloading medical datasets"):
        print("❌ Failed to download data. Please check your internet connection.")
        return
    
    # Step 3: Verify data structure
    print("\n📁 Verifying data structure...")
    data_dir = "data"
    qa_json_dir = os.path.join(data_dir, "QA_json")
    images_dir = os.path.join(data_dir, "images")
    
    required_files = [
        os.path.join(qa_json_dir, "train_vqa_data.json"),
        os.path.join(qa_json_dir, "valid_vqa_data.json"),
        os.path.join(qa_json_dir, "test_vqa_data.json")
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please run download_data.py manually to fix this.")
        return
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        print("Please run download_data.py manually to fix this.")
        return
    
    print("✅ Data structure verified!")
    
    # Step 4: Start training
    print("\n🎯 Starting SFT training...")
    print("This may take several hours depending on your GPU and dataset size.")
    
    user_input = input("\nDo you want to start training now? (y/n): ").strip().lower()
    
    if user_input == 'y':
        if not run_command("python train.py", "SFT Training"):
            print("❌ Training failed. Please check the error messages above.")
            return
        
        print("\n🎉 Training completed successfully!")
        print("\n📝 Next steps:")
        print("1. Test your model: python inference.py")
        print("2. Interactive testing: python inference.py --interactive")
        print(f"3. Find your trained model in: {os.path.join(os.getcwd(), 'medgemma4b_it_sft_reasoning')}")
        
    else:
        print("\n⏸️  Training skipped. You can start it later with:")
        print("   python train.py")
    
    print("\n✅ Setup completed successfully!")

if __name__ == "__main__":
    main()