import re
import torch
from typing import List, Dict, Any

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from config import *
from torch_dataloader import create_torch_dataloaders

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Create PyTorch DataLoaders (memory-efficient alternative to IterableDataset)
print("Creating PyTorch DataLoaders...")
print("💾 Images will be loaded on-demand during training")

train_loader, val_loader, test_loader = create_torch_dataloaders(
    train_json=TRAIN_JSON,
    val_json=VAL_JSON,
    test_json=TEST_JSON,
    batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
    num_workers=TRAINING_CONFIG.get("dataloader_num_workers", 0)
)

print("✅ PyTorch DataLoaders created successfully")

# Test dataloader by taking a sample
print("Testing dataloader...")
train_batch = next(iter(train_loader))
print(f"Sample batch keys: {list(train_batch.keys())}")
print(f"Batch size: {len(train_batch['prompt'])}")
print(f"Sample image sizes: {[img.size for img in train_batch['image']]}")

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

# --- Custom Dataset Wrapper for GRPOTrainer ---
class DataLoaderDataset:
    """
    Wrapper to make PyTorch DataLoader compatible with GRPOTrainer
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iterator = None
        self._length = None
    
    def __iter__(self):
        self._iterator = iter(self.dataloader)
        return self
    
    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            return next(self._iterator)
        except StopIteration:
            # Reset iterator for continuous iteration
            self._iterator = iter(self.dataloader)
            return next(self._iterator)
    
    def __len__(self):
        if self._length is None:
            # Estimate length based on dataset size and batch size
            try:
                dataset_size = len(self.dataloader.dataset)
                batch_size = self.dataloader.batch_size
                self._length = (dataset_size + batch_size - 1) // batch_size
            except:
                # Fallback if we can't determine length
                self._length = 1000  # Conservative estimate
        return self._length

# Wrap the dataloader for GRPOTrainer
train_dataset_wrapper = DataLoaderDataset(train_loader)

# --- GRPO Training Configuration ---
print("Setting up GRPO training configuration...")

training_args = GRPOConfig(
    learning_rate=TRAINING_CONFIG["learning_rate"],
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=TRAINING_CONFIG.get("logging_steps", 1),
    per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
    num_generations=TRAINING_CONFIG["num_generations"],
    num_train_epochs=TRAINING_CONFIG.get("num_train_epochs", 1),
    generation_batch_size=TRAINING_CONFIG["generation_batch_size"],
    max_steps=TRAINING_CONFIG.get("max_steps", 250),
    save_steps=TRAINING_CONFIG.get("save_steps", 250),
    max_grad_norm=0.1,
    report_to=TRAINING_CONFIG.get("report_to", "none"),
    output_dir=TRAINING_CONFIG["output_dir"],
    max_prompt_length=TRAINING_CONFIG.get("max_prompt_length"),
    max_completion_length=TRAINING_CONFIG.get("max_completion_length", 128),
    bf16=TRAINING_CONFIG.get("bf16", True),
    remove_unused_columns=TRAINING_CONFIG.get("remove_unused_columns", False),
    dataloader_num_workers=0,  # We handle this in our custom dataloader
    gradient_checkpointing=TRAINING_CONFIG.get("gradient_checkpointing", True),
    warmup_steps=TRAINING_CONFIG.get("warmup_steps", 50),
    beta=TRAINING_CONFIG.get("beta", 0.0),
    scale_rewards=TRAINING_CONFIG.get("scale_rewards", True),
)

print("Creating GRPO trainer...")
training_args.steps_per_generation = None

# Create trainer with PyTorch DataLoader
trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    args=training_args,
    train_dataset=train_dataset_wrapper,  # Use our wrapped dataloader
    reward_funcs=[reasoning_reward],
)

print("Starting GRPO training...")
print("🚀 Using PyTorch DataLoader for memory-efficient training")
trainer.train()

print("Saving trained model...")
trainer.save_model(TRAINING_CONFIG["output_dir"])
processor.save_pretrained(TRAINING_CONFIG["output_dir"])

print("Training completed!")