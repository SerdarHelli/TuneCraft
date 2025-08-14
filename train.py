import os
import re
import torch
from typing import List, Dict, Any
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from config import *

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create iterable datasets (following HuggingFace best practices)
print("Creating iterable datasets...")

from create_datasets import create_iterable_dataset

# Create datasets using the optimized iterable approach
ds_train = create_iterable_dataset(TRAIN_JSON)
ds_val = create_iterable_dataset(VAL_JSON)
ds_test = create_iterable_dataset(TEST_JSON)

print("✅ Iterable datasets created successfully")
print("💾 Images will be loaded on-demand during training")

# Test dataset by taking a few samples
print("Testing dataset...")
train_sample = next(iter(ds_train))
print(f"Sample columns: {list(train_sample.keys())}")
print(f"Sample prompt length: {len(train_sample['prompt'])}")
print(f"Sample image size: {train_sample['image'].size}")

LETTER_RE = re.compile(r"\b([ABCD])\b", flags=re.IGNORECASE)
FORMAT_RE = re.compile(r"^\s*([ABCD])\s*[-–:]\s*(.+)$", flags=re.IGNORECASE)

def between(text, start, end):
    try:
        s = text.index(start) + len(start)
        e = text.index(end, s)
        return text[s:e].strip()
    except ValueError:
        return ""

def parse_solution(text):
    sol = between(text, SOLUTION_START, SOLUTION_END)
    if not sol:
        return "", ""
    m = FORMAT_RE.match(sol)
    if not m:
        m2 = LETTER_RE.search(sol)
        return (m2.group(1).upper() if m2 else ""), sol.strip()
    return m.group(1).upper(), m.group(2).strip()

def parse_reasoning(text): return between(text, REASONING_START, REASONING_END)

def token_f1(pred, gold):
    import collections
    p = [t for t in re.findall(r"\w+", (pred or "").lower()) if t]
    g = [t for t in re.findall(r"\w+", (gold or "").lower()) if t]
    if not p or not g: return 0.0
    pc, gc = collections.Counter(p), collections.Counter(g)
    overlap = sum((pc & gc).values())
    if overlap == 0: return 0.0
    prec, rec = overlap/len(p), overlap/len(g)
    return 2*prec*rec/(prec+rec)

def reasoning_reward(prompts, completions, **kwargs):
    """
    Reward function for GRPO training.
    
    Args:
        prompts: List of prompt strings (not used directly but required by GRPO)
        completions: List of generated completion strings
        **kwargs: Additional columns from dataset (gold_letter, gold_explanation, etc.)
    
    Returns:
        List of reward scores (floats)
    """
    gold_letter      = kwargs.get("gold_letter", [])
    gold_explanation = kwargs.get("gold_explanation", [])
    impression       = kwargs.get("impression", [])
    findings         = kwargs.get("findings", [])
    indication       = kwargs.get("indication", [])
    heur_reason      = kwargs.get("heur_reason", [])

    scores = []
    for i, comp in enumerate(completions):
        text = comp if isinstance(comp, str) else str(comp)
        text = text.strip()
        
        # Extract reasoning and solution parts
        reasoning = parse_reasoning(text)
        letter, main_ans = parse_solution(text)

        score = 0.0
        
        # Reward correct answer (most important)
        gL = gold_letter[i] if i < len(gold_letter) else ""
        if letter and letter == (gL or "").upper(): 
            score += REWARD_WEIGHTS["correct_answer"]
        
        # Reward explanation quality
        gE = gold_explanation[i] if i < len(gold_explanation) else ""
        score += REWARD_WEIGHTS["explanation_quality"] * token_f1(main_ans, gE or "")

        # Reward use of evidence from report context
        def bag(x): return set(w for w in re.findall(r"\w+", (x or "").lower()) if len(w) > 3)
        
        imp = impression[i] if i < len(impression) else ""
        fin = findings[i] if i < len(findings) else ""
        ind = indication[i] if i < len(indication) else ""
        hr = heur_reason[i] if i < len(heur_reason) else ""
        
        evidence = bag(imp) | bag(fin) | bag(ind) | bag(hr)
        out_words = bag(reasoning) | bag(main_ans)
        if evidence and (evidence & out_words): 
            score += REWARD_WEIGHTS["evidence_usage"]

        # Reward proper format
        sol_text = between(text, SOLUTION_START, SOLUTION_END)
        if sol_text and FORMAT_RE.match(sol_text): 
            score += REWARD_WEIGHTS["format_bonus"]
            
        # Reward conciseness
        if len(text) <= 220: 
            score += REWARD_WEIGHTS["conciseness_bonus"]
            
        scores.append(score)
    
    return scores

# --- Model Setup ---
torch.backends.cuda.matmul.allow_tf32 = True

print(f"Loading model: {MODEL_ID}")

# Quantization config for memory efficiency
quant_config = BitsAndBytesConfig(
    load_in_4bit=QUANTIZATION_CONFIG["load_in_4bit"], 
    bnb_4bit_quant_type=QUANTIZATION_CONFIG["bnb_4bit_quant_type"],
    bnb_4bit_use_double_quant=QUANTIZATION_CONFIG["bnb_4bit_use_double_quant"], 
    bnb_4bit_compute_dtype=getattr(torch, QUANTIZATION_CONFIG["bnb_4bit_compute_dtype"])
)

# Load model - use AutoModelForImageTextToText for MedGemma VLM
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    quantization_config=quant_config, 
    device_map="auto",
    trust_remote_code=True
)

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(**LORA_CONFIG)
model = get_peft_model(model, lora_config)

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
    processor.tokenizer.padding_side = "left"

print("Model and processor loaded successfully")

# --- GRPO Training Configuration ---
print("Setting up GRPO training configuration...")

training_args = GRPOConfig(**TRAINING_CONFIG)

print("Creating GRPO trainer...")

# Create trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    reward_funcs=[reasoning_reward],
)

print("Starting GRPO training...")
trainer.train()

print("Saving trained model...")
trainer.save_model(TRAINING_CONFIG["output_dir"])
processor.save_pretrained(TRAINING_CONFIG["output_dir"])

print("Training completed!")
