"""
Lazy loading dataset implementation for large-scale medical image datasets
"""

import json
import os
from typing import Dict, Any, Optional
from datasets import Dataset
from PIL import Image
import numpy as np
from config import *

class LazyMedicalImageDataset:
    """
    Custom dataset class for lazy loading of medical images
    Optimized for 600k+ image datasets
    """
    
    def __init__(self, json_path: str, cache_size: int = 1000):
        """
        Initialize lazy dataset
        
        Args:
            json_path: Path to JSON file with dataset
            cache_size: Number of images to keep in memory cache
        """
        self.json_path = json_path
        self.cache_size = cache_size
        self.image_cache = {}  # LRU cache for images
        self.cache_order = []  # Track access order for LRU
        
        # Load metadata only (not images)
        print(f"Loading metadata from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        self.samples = []
        for _id, e in raw_data.items():
            if not e.get("question") or not e.get("options") or not e.get("ImagePath"):
                continue
            
            image_paths = list(e["ImagePath"] or [])
            if USE_ONLY_FIRST_IMAGE and image_paths:
                image_paths = image_paths[:1]
            
            primary_image_path = image_paths[0] if image_paths else None
            
            self.samples.append({
                "id": _id,
                "prompt": self._build_prompt_text(e),
                "image_path": primary_image_path,
                "gold_letter": self._safe_text(e.get("correct_answer")),
                "gold_explanation": self._safe_text(e.get("correct_answer_explanation")),
                "impression": self._safe_text(e.get("Impression")),
                "findings": self._safe_text(e.get("Findings")),
                "indication": self._safe_text(e.get("Indication")),
                "heur_reason": self._safe_text(e.get("reason")),
                "expected_response": self._build_expected_response(e),
            })
        
        print(f"Loaded {len(self.samples)} samples with lazy image loading")
    
    def _safe_text(self, x) -> str:
        """Convert to safe text string"""
        if x is None:
            return ""
        try:
            return str(x).strip()
        except Exception:
            pass
        return str(x)
    
    def _build_prompt_text(self, e: Dict[str, Any]) -> str:
        """Build the prompt text for GRPO training"""
        opts = "\n".join(e["options"])
        ctx = []
        if e.get("Indication"): ctx.append(f"Indication: {self._safe_text(e['Indication'])}")
        if e.get("Comparison"): ctx.append(f"Comparison: {self._safe_text(e['Comparison'])}")
        if e.get("Findings"): ctx.append(f"Findings: {self._safe_text(e['Findings'])}")
        if e.get("Impression"): ctx.append(f"Impression: {self._safe_text(e['Impression'])}")
        if e.get("reason"): ctx.append(f"Heuristic reason: {self._safe_text(e['reason'])}")
        
        return (
            "You are an expert radiologist. Use the chest X-ray image(s) and report context.\n"
            "Respond in this format:\n"
            f"  {REASONING_START}...{REASONING_END}{SOLUTION_START}LETTER - main answer{SOLUTION_END}\n\n"
            f"Question:\n{self._safe_text(e['question'])}\n\nOptions:\n{opts}\n\nReport Context:\n" +
            ("\n".join(ctx) if ctx else "N/A")
        )
    
    def _build_expected_response(self, e: Dict[str, Any]) -> str:
        """Build the expected response for reference"""
        rsn_parts = []
        if e.get("Indication"): rsn_parts.append(f"Indication: {self._safe_text(e['Indication'])}")
        if e.get("Findings"): rsn_parts.append(f"Findings: {self._safe_text(e['Findings'])}")
        if e.get("Impression"): rsn_parts.append(f"Impression: {self._safe_text(e['Impression'])}")
        if e.get("reason"): rsn_parts.append(f"Heuristic reason: {self._safe_text(e['reason'])}")
        
        reasoning = f"{REASONING_START}" + " ".join(rsn_parts).strip() + f"{REASONING_END}"
        solution = f"{SOLUTION_START}{self._safe_text(e.get('correct_answer'))} - {self._safe_text(e.get('correct_answer_explanation'))}{SOLUTION_END}"
        return reasoning + solution
    
    def _load_and_cache_image(self, image_path: str, index: int) -> Image.Image:
        """Load image with LRU caching"""
        # Check if image is in cache
        if index in self.image_cache:
            # Move to end of cache order (most recently used)
            self.cache_order.remove(index)
            self.cache_order.append(index)
            return self.image_cache[index]
        
        # Load image
        if image_path and os.path.exists(image_path):
            image = self._load_and_preprocess_medical_image(image_path)
        else:
            image = Image.new("RGB", IMAGE_CONFIG["target_size"], color=(128, 128, 128))
        
        # Add to cache
        self.image_cache[index] = image
        self.cache_order.append(index)
        
        # Remove oldest items if cache is full
        while len(self.image_cache) > self.cache_size:
            oldest_index = self.cache_order.pop(0)
            del self.image_cache[oldest_index]
        
        return image
    
    def _load_and_preprocess_medical_image(self, image_path: str) -> Image.Image:
        """Load and preprocess medical images"""
        try:
            # Load image
            img = Image.open(image_path)
            
            # Handle different image modes
            if img.mode in ['I', 'I;16', 'F']:  # 16-bit or float images
                # Convert to numpy for processing
                img_array = np.array(img, dtype=np.float32)
                
                # Normalize to 0-255 range
                if img_array.max() > 255:
                    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255
                
                # Convert to 8-bit
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
            
            # Resize image
            if img.size != IMAGE_CONFIG["target_size"]:
                img = img.resize(IMAGE_CONFIG["target_size"], Image.Resampling.LANCZOS)
            
            return img
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return Image.new("RGB", IMAGE_CONFIG["target_size"], color=(128, 128, 128))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """Get item with lazy image loading"""
        sample = self.samples[index].copy()
        
        # Load image on demand
        image_path = sample.pop("image_path")
        sample["image"] = self._load_and_cache_image(image_path, index)
        
        return sample
    
    def to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace Dataset with lazy loading"""
        def generate_samples():
            for i in range(len(self)):
                yield self[i]
        
        # Create dataset from generator (memory efficient)
        dataset = Dataset.from_generator(generate_samples)
        
        # Set transform for lazy loading
        def lazy_transform(examples):
            # This transform will be applied on-the-fly during training
            if isinstance(examples, dict):
                # Single example
                if "image_path" in examples:
                    examples["image"] = self._load_and_cache_image(examples["image_path"], 0)
                return examples
            else:
                # Batch of examples
                for i, example in enumerate(examples):
                    if "image_path" in example:
                        example["image"] = self._load_and_cache_image(example["image_path"], i)
                return examples
        
        return dataset

def create_lazy_datasets(train_json: str, val_json: str, test_json: str, cache_size: int = 1000):
    """Create lazy loading datasets for all splits"""
    print("Creating lazy loading datasets...")
    
    train_dataset = LazyMedicalImageDataset(train_json, cache_size=cache_size)
    val_dataset = LazyMedicalImageDataset(val_json, cache_size=cache_size//2)  # Smaller cache for val
    test_dataset = LazyMedicalImageDataset(test_json, cache_size=cache_size//2)
    
    # Convert to HuggingFace datasets
    ds_train = train_dataset.to_hf_dataset()
    ds_val = val_dataset.to_hf_dataset()
    ds_test = test_dataset.to_hf_dataset()
    
    return ds_train, ds_val, ds_test