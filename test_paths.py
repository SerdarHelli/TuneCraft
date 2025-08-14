"""
Test script to check image path resolution
"""
import os
import json
from config import *

def test_path_resolution():
    """Test how image paths are resolved"""
    print("🔍 Testing Image Path Resolution")
    print("=" * 50)
    
    # Test with training JSON
    if os.path.exists(TRAIN_JSON):
        print(f"📁 Testing with: {TRAIN_JSON}")
        
        with open(TRAIN_JSON, 'r') as f:
            data = json.load(f)
        
        # Handle both dict and list formats
        if isinstance(data, dict):
            items = list(data.values())[:5]  # Test first 5 items
        else:
            items = data[:5]
        
        print(f"\n🔍 Testing first 5 image paths:")
        print(f"Current working directory: {os.getcwd()}")
        
        for i, item in enumerate(items, 1):
            print(f"\n--- Sample {i} ---")
            
            # Get raw path from JSON
            raw_path = item.get('image_path', 'NOT_FOUND')
            print(f"Raw path from JSON: {raw_path}")
            
            # Apply path resolution logic
            if raw_path.startswith('../'):
                resolved_path = os.path.normpath(os.path.join(os.getcwd(), raw_path))
                print(f"Resolved path: {resolved_path}")
            else:
                resolved_path = os.path.join("data/images", raw_path)
                print(f"Standard path: {resolved_path}")
            
            # Check if file exists
            exists = os.path.exists(resolved_path)
            print(f"File exists: {'✅ YES' if exists else '❌ NO'}")
            
            if not exists:
                # Try to find where the file might be
                filename = os.path.basename(raw_path)
                print(f"Looking for filename: {filename}")
                
                # Check common locations
                possible_paths = [
                    os.path.join("data", "images", filename),
                    os.path.join("data", filename),
                    os.path.join("..", "data", "images", filename),
                    filename
                ]
                
                for possible_path in possible_paths:
                    if os.path.exists(possible_path):
                        print(f"Found at: {os.path.abspath(possible_path)}")
                        break
                else:
                    print("File not found in common locations")
    
    else:
        print(f"❌ Training JSON not found: {TRAIN_JSON}")
        print("Please run download_data.py first")

def show_directory_structure():
    """Show current directory structure to help debug paths"""
    print(f"\n📂 Current Directory Structure:")
    print(f"Working directory: {os.getcwd()}")
    
    # Show what's in current directory
    print("\nFiles in current directory:")
    for item in sorted(os.listdir(".")):
        if os.path.isdir(item):
            print(f"  📁 {item}/")
        else:
            print(f"  📄 {item}")
    
    # Check for data directory
    if os.path.exists("data"):
        print(f"\nFiles in data/ directory:")
        for item in sorted(os.listdir("data")):
            if os.path.isdir(os.path.join("data", item)):
                print(f"  📁 data/{item}/")
            else:
                print(f"  📄 data/{item}")
        
        # Check for images subdirectory
        if os.path.exists("data/images"):
            image_count = len([f for f in os.listdir("data/images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"\n📸 Found {image_count} images in data/images/")
    
    # Check parent directory for data
    parent_data = os.path.join("..", "data")
    if os.path.exists(parent_data):
        print(f"\n📁 Found data directory in parent: {os.path.abspath(parent_data)}")
        if os.path.exists(os.path.join(parent_data, "images")):
            image_count = len([f for f in os.listdir(os.path.join(parent_data, "images")) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"📸 Found {image_count} images in ../data/images/")

if __name__ == "__main__":
    show_directory_structure()
    test_path_resolution()