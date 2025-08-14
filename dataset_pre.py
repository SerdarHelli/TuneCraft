import json
import os
import re
from typing import Dict, Any, List
from datasets import Dataset
from PIL import Image
import torch
import numpy as np
from config import *

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

def safe_text(x):
    if x is None: return ""
    try:
        from math import isnan
        if isinstance(x, float) and isnan(x): return ""
    except Exception:
        pass
    return str(x)

def build_prompt_text(e: Dict[str, Any]) -> str:
    """Build the prompt text for GRPO training (prompt-only format)"""
    opts = "\n".join(e["options"])
    ctx  = []
    if e.get("Indication"): ctx.append(f"Indication: {safe_text(e['Indication'])}")
    if e.get("Comparison"): ctx.append(f"Comparison: {safe_text(e['Comparison'])}")
    if e.get("Findings"):   ctx.append(f"Findings: {safe_text(e['Findings'])}")
    if e.get("Impression"): ctx.append(f"Impression: {safe_text(e['Impression'])}")
    if e.get("reason"):     ctx.append(f"Heuristic reason: {safe_text(e['reason'])}")
    return (
        "You are an expert radiologist. Use the chest X-ray image(s) and report context.\n"
        "Respond in this format:\n"
        f"  {REASONING_START}...{REASONING_END}{SOLUTION_START}LETTER - main answer{SOLUTION_END}\n\n"
        f"Question:\n{safe_text(e['question'])}\n\nOptions:\n{opts}\n\nReport Context:\n" +
        ("\n".join(ctx) if ctx else "N/A")
    )

def build_expected_response(e: Dict[str, Any]) -> str:
    """Build the expected response for reference (not used in GRPO training directly)"""
    rsn_parts: List[str] = []
    if e.get("Indication"): rsn_parts.append(f"Indication: {safe_text(e['Indication'])}")
    if e.get("Findings"):   rsn_parts.append(f"Findings: {safe_text(e['Findings'])}")
    if e.get("Impression"): rsn_parts.append(f"Impression: {safe_text(e['Impression'])}")
    if e.get("reason"):     rsn_parts.append(f"Heuristic reason: {safe_text(e['reason'])}")
    reasoning = f"{REASONING_START}" + " ".join(rsn_parts).strip() + f"{REASONING_END}"
    solution  = f"{SOLUTION_START}{safe_text(e.get('correct_answer'))} - {safe_text(e.get('correct_answer_explanation'))}{SOLUTION_END}"
    return reasoning + solution

def load_and_preprocess_medical_image(image_path: str, target_size: tuple = (512, 512)) -> Image.Image:
    """
    Load and preprocess medical images for MedGemma-4B
    - Handles 16-bit medical images (DICOM, PNG, etc.)
    - Converts to 8-bit RGB
    - Resizes to target size
    - Applies proper normalization
    """
    try:
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
            return Image.new("RGB", target_size, color=(128, 128, 128))
        
        # Load image
        img = Image.open(image_path)
        
        # Handle different image modes
        if img.mode in ['I', 'I;16', 'F']:  # 16-bit or float images
            print(f"Converting 16-bit image: {image_path}")
            # Convert to numpy for processing
            import numpy as np
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to 0-255 range
            if img_array.max() > 255:
                # For 16-bit images, normalize from full range
                img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
            
            # Convert to 8-bit
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # Create PIL image from array
            img = Image.fromarray(img_array, mode='L')
        
        # Convert to RGB (required for vision models)
        if img.mode != 'RGB':
            if img.mode == 'L':  # Grayscale
                # Convert grayscale to RGB by duplicating channels
                img = img.convert('RGB')
            elif img.mode in ['RGBA', 'LA']:  # With alpha channel
                # Create white background and paste image
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])  # Use alpha as mask
                else:
                    background.paste(img.convert('RGB'))
                img = background
            else:
                img = img.convert('RGB')
        
        # Resize image to target size
        if img.size != target_size:
            # Use LANCZOS for high-quality downsampling
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        return img
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        # Return placeholder image
        return Image.new("RGB", target_size, color=(128, 128, 128))

def build_dataset_from_json(json_path: str) -> Dataset:
    """Build dataset in GRPO format with lazy image loading"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for _id, e in raw.items():
        if not e.get("question") or not e.get("options") or not e.get("ImagePath"):
            continue
        
        image_paths = list(e["ImagePath"] or [])
        if USE_ONLY_FIRST_IMAGE and image_paths:
            image_paths = image_paths[:1]
        
        # Store image path instead of loading image immediately (lazy loading)
        primary_image_path = image_paths[0] if image_paths else None

        rows.append({
            "id": _id,
            "prompt": build_prompt_text(e),  # Required for GRPO
            "image_path": primary_image_path,  # Store path for lazy loading
            # Additional columns for reward function
            "gold_letter": safe_text(e.get("correct_answer")),
            "gold_explanation": safe_text(e.get("correct_answer_explanation")),
            "impression": safe_text(e.get("Impression")),
            "findings": safe_text(e.get("Findings")),
            "indication": safe_text(e.get("Indication")),
            "heur_reason": safe_text(e.get("reason")),
            "expected_response": build_expected_response(e),  # For reference
        })

    # Create dataset with image paths
    dataset = Dataset.from_list(rows)
    
    # Add lazy image loading transform
    def load_image_on_demand(example):
        """Load and preprocess image only when accessed"""
        if example["image_path"]:
            example["image"] = load_and_preprocess_medical_image(
                example["image_path"], 
                target_size=IMAGE_CONFIG["target_size"]
            )
        else:
            example["image"] = Image.new("RGB", IMAGE_CONFIG["target_size"], color=(128, 128, 128))
        return example
    
    # Apply transform but don't load all images at once
    dataset.set_transform(load_image_on_demand)
    
    return dataset

print("Building datasets from JSON files with lazy loading...")
print(f"Train JSON: {TRAIN_JSON}")
print(f"Val JSON: {VAL_JSON}")
print(f"Test JSON: {TEST_JSON}")

# For large datasets (600k images), use lazy loading
from lazy_dataset import create_lazy_datasets

# Create lazy datasets with image caching
# Cache size: number of images to keep in memory
# Use cache size from config
CACHE_SIZE = LAZY_LOADING_CONFIG["cache_size"]

print("Creating lazy loading datasets...")
ds_train, ds_val, ds_test = create_lazy_datasets(
    TRAIN_JSON, VAL_JSON, TEST_JSON, 
    cache_size=CACHE_SIZE
)

print(f"Train dataset: {len(ds_train)} samples")
print(f"Val dataset: {len(ds_val)} samples") 
print(f"Test dataset: {len(ds_test)} samples")

print(f"Saving datasets with lazy loading...")
ds_train.save_to_disk(DATASET_SAVE_NAMES["train"])
ds_val.save_to_disk(DATASET_SAVE_NAMES["val"])
ds_test.save_to_disk(DATASET_SAVE_NAMES["test"])

print("✅ Datasets created with lazy image loading!")
print(f"💾 Memory usage: Only ~{CACHE_SIZE} images cached at a time")
print(f"🚀 Ready for training with 600k+ images!")

print(ds_train)
print(ds_val)
print(ds_test)
