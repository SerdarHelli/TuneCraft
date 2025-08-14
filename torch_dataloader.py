#!/usr/bin/env python3
"""
PyTorch DataLoader implementation for large-scale medical VQA datasets
Memory-efficient alternative to HuggingFace IterableDataset
"""

import json
import os
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from config import *

class MedicalVQADataset(Dataset):
    """
    PyTorch Dataset for Medical VQA with on-demand image loading
    """
    
    def __init__(self, json_path: str, transform=None):
        """
        Initialize the dataset
        
        Args:
            json_path: Path to the JSON file containing the data
            transform: Optional image transforms
        """
        self.json_path = json_path
        self.transform = transform
        
        # Load the JSON data
        print(f"Loading data from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        
        # Filter and prepare data indices
        self.data_items = []
        for _id, e in raw_data.items():
            if not e.get("question") or not e.get("options") or not e.get("ImagePath"):
                continue
            
            image_paths = list(e["ImagePath"] or [])
            if USE_ONLY_FIRST_IMAGE and image_paths:
                image_paths = image_paths[:1]
            
            primary_image_path = image_paths[0] if image_paths else None
            
            self.data_items.append({
                "id": _id,
                "data": e,
                "image_path": primary_image_path
            })
        
        print(f"Loaded {len(self.data_items)} valid samples from {json_path}")
    
    def __len__(self) -> int:
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset
        Images are loaded on-demand to save memory
        """
        item = self.data_items[idx]
        data = item["data"]
        image_path = item["image_path"]
        
        # Load image on-demand
        if image_path and os.path.exists(image_path):
            image = self.load_and_preprocess_medical_image(image_path)
        else:
            # Create placeholder image if path doesn't exist
            image = Image.new("RGB", IMAGE_CONFIG["target_size"], color=(128, 128, 128))
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Build prompt
        prompt = self.build_prompt_text(data)
        
        return {
            "id": item["id"],
            "prompt": prompt,
            "image": image,
            "gold_letter": self.safe_text(data.get("correct_answer")),
            "gold_explanation": self.safe_text(data.get("correct_answer_explanation")),
            "impression": self.safe_text(data.get("Impression")),
            "findings": self.safe_text(data.get("Findings")),
            "indication": self.safe_text(data.get("Indication")),
            "heur_reason": self.safe_text(data.get("reason")),
        }
    
    def safe_text(self, x) -> str:
        """Convert to safe text string"""
        if x is None:
            return ""
        try:
            return str(x).strip()
        except Exception:
            pass
        return str(x)
    
    def build_prompt_text(self, e: dict) -> str:
        """Build the prompt text for GRPO training"""
        opts = "\n".join(e["options"])
        ctx = []
        if e.get("Indication"): ctx.append(f"Indication: {self.safe_text(e['Indication'])}")
        if e.get("Comparison"): ctx.append(f"Comparison: {self.safe_text(e['Comparison'])}")
        if e.get("Findings"): ctx.append(f"Findings: {self.safe_text(e['Findings'])}")
        if e.get("Impression"): ctx.append(f"Impression: {self.safe_text(e['Impression'])}")
        if e.get("reason"): ctx.append(f"Heuristic reason: {self.safe_text(e['reason'])}")
        
        return (
            "You are an expert radiologist. Use the chest X-ray image(s) and report context.\n"
            "Respond in this format:\n"
            f"  {REASONING_START}...{REASONING_END}{SOLUTION_START}LETTER - main answer{SOLUTION_END}\n\n"
            f"Question:\n{self.safe_text(e['question'])}\n\nOptions:\n{opts}\n\nReport Context:\n" +
            ("\n".join(ctx) if ctx else "N/A")
        )
    
    def load_and_preprocess_medical_image(self, image_path: str) -> Image.Image:
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


class MedicalVQADataModule:
    """
    DataModule for managing train/val/test dataloaders
    """
    
    def __init__(
        self,
        train_json: str,
        val_json: str,
        test_json: str,
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False
    ):
        self.train_json = train_json
        self.val_json = val_json
        self.test_json = test_json
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """Setup datasets"""
        print("Setting up datasets...")
        self.train_dataset = MedicalVQADataset(self.train_json)
        self.val_dataset = MedicalVQADataset(self.val_json)
        self.test_dataset = MedicalVQADataset(self.test_json)
        print("✅ Datasets setup complete")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        if self.train_dataset is None:
            self.setup()
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,  # Important for consistent batch sizes
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        if self.val_dataset is None:
            self.setup()
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader"""
        if self.test_dataset is None:
            self.setup()
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=self.collate_fn
        )
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function to handle batching
        """
        # Separate different types of data
        ids = [item["id"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        images = [item["image"] for item in batch]
        gold_letters = [item["gold_letter"] for item in batch]
        gold_explanations = [item["gold_explanation"] for item in batch]
        impressions = [item["impression"] for item in batch]
        findings = [item["findings"] for item in batch]
        indications = [item["indication"] for item in batch]
        heur_reasons = [item["heur_reason"] for item in batch]
        
        return {
            "id": ids,
            "prompt": prompts,
            "image": images,
            "gold_letter": gold_letters,
            "gold_explanation": gold_explanations,
            "impression": impressions,
            "findings": findings,
            "indication": indications,
            "heur_reason": heur_reasons,
        }


def create_torch_dataloaders(
    train_json: str,
    val_json: str,
    test_json: str,
    batch_size: int = 2,
    num_workers: int = 0
) -> tuple:
    """
    Create PyTorch DataLoaders for training
    
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    datamodule = MedicalVQADataModule(
        train_json=train_json,
        val_json=val_json,
        test_json=test_json,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    return train_loader, val_loader, test_loader


def test_dataloader():
    """Test the dataloader implementation"""
    print("🧪 Testing PyTorch DataLoader implementation...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_torch_dataloaders(
        train_json=TRAIN_JSON,
        val_json=VAL_JSON,
        test_json=TEST_JSON,
        batch_size=2,
        num_workers=0
    )
    
    # Test train loader
    print("Testing train dataloader...")
    train_batch = next(iter(train_loader))
    print(f"  ✅ Batch keys: {list(train_batch.keys())}")
    print(f"  ✅ Batch size: {len(train_batch['prompt'])}")
    print(f"  ✅ Image sizes: {[img.size for img in train_batch['image']]}")
    print(f"  ✅ Prompt lengths: {[len(p) for p in train_batch['prompt']]}")
    
    print("\n🎉 PyTorch DataLoader test successful!")
    print("💡 Key benefits:")
    print("   • Memory efficient on-demand image loading")
    print("   • Full control over batching and data loading")
    print("   • Compatible with any PyTorch-based trainer")
    print("   • Supports multiprocessing for faster loading")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    test_dataloader()