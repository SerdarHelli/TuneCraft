#!/usr/bin/env python3
"""
Create iterable datasets following HuggingFace best practices for large datasets
"""

import json
import os
from datasets import IterableDataset
from PIL import Image
import numpy as np
from config import *

def safe_text(x) -> str:
    """Convert to safe text string"""
    if x is None:
        return ""
    try:
        return str(x).strip()
    except Exception:
        pass
    return str(x)

def build_prompt_text(e: dict) -> str:
    """Build the prompt text for GRPO training"""
    opts = "\n".join(e["options"])
    ctx = []
    if e.get("Indication"): ctx.append(f"Indication: {safe_text(e['Indication'])}")
    if e.get("Comparison"): ctx.append(f"Comparison: {safe_text(e['Comparison'])}")
    if e.get("Findings"): ctx.append(f"Findings: {safe_text(e['Findings'])}")
    if e.get("Impression"): ctx.append(f"Impression: {safe_text(e['Impression'])}")
    if e.get("reason"): ctx.append(f"Heuristic reason: {safe_text(e['reason'])}")
    
    return (
        "You are an expert radiologist. Use the chest X-ray image(s) and report context.\n"
        "Respond in this format:\n"
        f"  {REASONING_START}...{REASONING_END}{SOLUTION_START}LETTER - main answer{SOLUTION_END}\n\n"
        f"Question:\n{safe_text(e['question'])}\n\nOptions:\n{opts}\n\nReport Context:\n" +
        ("\n".join(ctx) if ctx else "N/A")
    )

def load_and_preprocess_medical_image(image_path: str) -> Image.Image:
    """Load and preprocess medical images for MedGemma-4B"""
    try:
        if not os.path.exists(image_path):
            return Image.new("RGB", IMAGE_CONFIG["target_size"], color=(128, 128, 128))
        
        # Load image
        img = Image.open(image_path)
        
        # Handle 16-bit images
        if img.mode in ['I', 'I;16', 'F']:
            img_array = np.array(img, dtype=np.float32)
            if img_array.max() > 255:
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
        
        # Convert to RGB
        if img.mode != 'RGB':
            if img.mode == 'L':
                img = img.convert('RGB')
            elif img.mode in ['RGBA', 'LA']:
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img.convert('RGB'))
                img = background
            else:
                img = img.convert('RGB')
        
        # Resize
        if img.size != IMAGE_CONFIG["target_size"]:
            img = img.resize(IMAGE_CONFIG["target_size"], Image.Resampling.LANCZOS)
        
        return img
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return Image.new("RGB", IMAGE_CONFIG["target_size"], color=(128, 128, 128))

def create_data_generator(json_path: str):
    """Create a generator function for the dataset"""
    def generator():
        print(f"Loading data from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        print(f"Processing {len(raw_data)} samples...")
        processed = 0
        
        for _id, e in raw_data.items():
            if not e.get("question") or not e.get("options") or not e.get("ImagePath"):
                continue
            
            image_paths = list(e["ImagePath"] or [])
            if USE_ONLY_FIRST_IMAGE and image_paths:
                image_paths = image_paths[:1]
            
            primary_image_path = image_paths[0] if image_paths else None
            
            # Load image on-demand (this is the key for memory efficiency)
            if primary_image_path:
                image = load_and_preprocess_medical_image(primary_image_path)
            else:
                image = Image.new("RGB", IMAGE_CONFIG["target_size"], color=(128, 128, 128))
            
            yield {
                "id": _id,
                "prompt": build_prompt_text(e),
                "image": image,  # Image loaded here, not stored as path
                "gold_letter": safe_text(e.get("correct_answer")),
                "gold_explanation": safe_text(e.get("correct_answer_explanation")),
                "impression": safe_text(e.get("Impression")),
                "findings": safe_text(e.get("Findings")),
                "indication": safe_text(e.get("Indication")),
                "heur_reason": safe_text(e.get("reason")),
            }
            
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed} samples...")
    
    return generator

def create_iterable_dataset(json_path: str) -> IterableDataset:
    """Create an iterable dataset from JSON file"""
    generator_fn = create_data_generator(json_path)
    
    # Create iterable dataset
    dataset = IterableDataset.from_generator(generator_fn)
    
    return dataset

def main():
    """Create all datasets"""
    print("🚀 Creating Iterable Datasets for Large-Scale Training")
    print("=" * 60)
    
    print(f"📁 Dataset paths:")
    print(f"  Train: {TRAIN_JSON}")
    print(f"  Val:   {VAL_JSON}")
    print(f"  Test:  {TEST_JSON}")
    
    print(f"\n🖼️  Image settings:")
    print(f"  Target size: {IMAGE_CONFIG['target_size']}")
    print(f"  Use only first image: {USE_ONLY_FIRST_IMAGE}")
    
    # Create datasets
    print(f"\n📊 Creating datasets...")
    
    print(f"\n1️⃣  Creating training dataset...")
    ds_train = create_iterable_dataset(TRAIN_JSON)
    
    print(f"\n2️⃣  Creating validation dataset...")
    ds_val = create_iterable_dataset(VAL_JSON)
    
    print(f"\n3️⃣  Creating test dataset...")
    ds_test = create_iterable_dataset(TEST_JSON)
    
    # Test datasets
    print(f"\n🧪 Testing datasets...")
    
    print("Testing train dataset...")
    train_sample = next(iter(ds_train))
    print(f"  ✅ Train sample keys: {list(train_sample.keys())}")
    print(f"  ✅ Image size: {train_sample['image'].size}")
    print(f"  ✅ Prompt length: {len(train_sample['prompt'])} chars")
    
    print("Testing val dataset...")
    val_sample = next(iter(ds_val))
    print(f"  ✅ Val sample keys: {list(val_sample.keys())}")
    
    print(f"\n🎉 Success! Iterable datasets created.")
    print(f"💡 Key benefits:")
    print(f"   • Images loaded on-demand (memory efficient)")
    print(f"   • Supports datasets of any size (600k+ images)")
    print(f"   • Automatic 16-bit to 8-bit conversion")
    print(f"   • No disk caching needed (generated fresh each time)")
    
    print(f"\n🚀 Ready for GRPO training!")
    
    return ds_train, ds_val, ds_test

if __name__ == "__main__":
    main()