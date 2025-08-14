import os
import shutil
import tarfile
import glob
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from datasets import Dataset

REXVQA_REPO = "rajpurkarlab/ReXVQA"
REXGRAD_REPO = "rajpurkarlab/ReXGradient-160K"

def download_and_extract_data():
    """Download and extract medical VQA datasets"""
    
    # Create necessary directories
    data_dir = Path("data")
    qa_json_dir = data_dir / "QA_json"
    images_dir = data_dir / "images"
    
    # Create directories if they don't exist
    data_dir.mkdir(exist_ok=True)
    qa_json_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    print("Downloading ReXGradient-160K dataset...")
    # Download ReXGradient dataset
    rexgrad_path = snapshot_download(repo_id=REXGRAD_REPO, repo_type="dataset")
    print(f"ReXGradient dataset downloaded to: {rexgrad_path}")
    
    # Combine and extract tar parts for images
    print("Extracting image data...")
    tar_parts = glob.glob(os.path.join(rexgrad_path, "deid_png.part*"))
    
    if tar_parts:
        # Sort parts to ensure correct order
        tar_parts.sort()
        
        # Combine tar parts
        combined_tar_path = data_dir / "deid_png.tar"
        with open(combined_tar_path, 'wb') as outfile:
            for part_file in tar_parts:
                print(f"Processing part: {part_file}")
                with open(part_file, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
        
        # Extract tar file
        print(f"Extracting tar file to {images_dir}")
        with tarfile.open(combined_tar_path, 'r') as tar:
            tar.extractall(path=images_dir)
        
        # Clean up tar file
        combined_tar_path.unlink()
        print("Image extraction completed!")
    else:
        print("No tar parts found in ReXGradient dataset")
    
    print("Downloading ReXVQA dataset...")
    # Download ReXVQA dataset
    rexvqa_path = snapshot_download(repo_id=REXVQA_REPO, repo_type="dataset")
    print(f"ReXVQA dataset downloaded to: {rexvqa_path}")
    
    # Copy metadata JSON files
    print("Copying metadata files...")
    metadata_files = [
        "test_vqa_data.json",
        "train_vqa_data.json", 
        "valid_vqa_data.json"
    ]
    
    metadata_source_dir = Path(rexvqa_path) / "metadata"
    
    for filename in metadata_files:
        source_file = metadata_source_dir / filename
        dest_file = qa_json_dir / filename
        
        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"Copied {filename} to {dest_file}")
        else:
            print(f"Warning: {filename} not found in {metadata_source_dir}")
    
    print("\nData download and extraction completed!")
    print(f"Images extracted to: {images_dir}")
    print(f"JSON metadata files copied to: {qa_json_dir}")
    
    # Print summary of downloaded data
    print("\nSummary:")
    if images_dir.exists():
        image_count = len(list(images_dir.rglob("*.png"))) + len(list(images_dir.rglob("*.jpg")))
        print(f"- Images directory: {image_count} image files")
    
    if qa_json_dir.exists():
        json_files = list(qa_json_dir.glob("*.json"))
        print(f"- JSON files: {len(json_files)} files")
        for json_file in json_files:
            print(f"  - {json_file.name}")

if __name__ == "__main__":
    download_and_extract_data()
