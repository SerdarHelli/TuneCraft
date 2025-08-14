#!/usr/bin/env python3
"""
Inference script for the trained MedGemma GRPO model
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np
import argparse
from config import *

def load_trained_model(model_path: str):
    """Load the trained model and processor"""
    print(f"Loading trained model from: {model_path}")
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    return model, processor

def load_and_preprocess_medical_image(image_path: str, target_size: tuple = (512, 512)) -> Image.Image:
    """Load and preprocess medical images (same as in dataset_pre.py)"""
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

def generate_response(model, processor, prompt: str, image_path: str, max_length: int = 128):
    """Generate a response for the given prompt and image"""
    
    # Load and process image using the same preprocessing as training
    image = load_and_preprocess_medical_image(image_path, target_size=IMAGE_CONFIG["target_size"])
    
    # Prepare inputs
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode response
    response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    return response

def create_sample_prompt(question: str, options: list, context: str = ""):
    """Create a prompt in the expected format"""
    opts = "\n".join(options)
    
    prompt = (
        "You are an expert radiologist. Use the chest X-ray image(s) and report context.\n"
        "Respond in this format:\n"
        f"  {REASONING_START}...{REASONING_END}{SOLUTION_START}LETTER - main answer{SOLUTION_END}\n\n"
        f"Question:\n{question}\n\nOptions:\n{opts}\n\nReport Context:\n{context or 'N/A'}"
    )
    
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained MedGemma GRPO model")
    parser.add_argument("--model_path", type=str, default=TRAINING_CONFIG["output_dir"],
                       help="Path to the trained model")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the chest X-ray image")
    parser.add_argument("--question", type=str, required=True,
                       help="Medical question to ask")
    parser.add_argument("--options", type=str, nargs="+", required=True,
                       help="Multiple choice options (e.g., 'A. Normal' 'B. Pneumonia')")
    parser.add_argument("--context", type=str, default="",
                       help="Additional clinical context")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum generation length")
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_trained_model(args.model_path)
    
    # Create prompt
    prompt = create_sample_prompt(args.question, args.options, args.context)
    
    print("=== Input ===")
    print(f"Image: {args.image_path}")
    print(f"Question: {args.question}")
    print(f"Options: {args.options}")
    print(f"Context: {args.context}")
    
    print("\n=== Prompt ===")
    print(prompt)
    
    # Generate response
    print("\n=== Generating Response ===")
    response = generate_response(model, processor, prompt, args.image_path, args.max_length)
    
    print("\n=== Response ===")
    print(response)
    
    # Try to parse the response
    print("\n=== Parsed Response ===")
    try:
        if REASONING_START in response and REASONING_END in response:
            reasoning = response.split(REASONING_START)[1].split(REASONING_END)[0].strip()
            print(f"Reasoning: {reasoning}")
        
        if SOLUTION_START in response and SOLUTION_END in response:
            solution = response.split(SOLUTION_START)[1].split(SOLUTION_END)[0].strip()
            print(f"Solution: {solution}")
            
            # Extract letter
            import re
            letter_match = re.search(r'\b([ABCD])\b', solution)
            if letter_match:
                print(f"Predicted Answer: {letter_match.group(1)}")
    except Exception as e:
        print(f"Could not parse response: {e}")

if __name__ == "__main__":
    main()