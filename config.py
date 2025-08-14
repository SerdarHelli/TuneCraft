"""
Configuration file for MedGemma GRPO training
"""

# Dataset paths - UPDATE THESE TO YOUR ACTUAL PATHS
TRAIN_JSON = "/home/QA_json/train_vqa_data.json"
VAL_JSON = "/home/QA_json/valid_vqa_data.json"
TEST_JSON = "/home/QA_json/test_vqa_data.json"

# Model configuration
MODEL_ID = "google/medgemma-4b-it"  # Medical vision-language model
USE_ONLY_FIRST_IMAGE = False  # Set to True to use only the first image per sample

# Training configuration - Optimized for A100 80GB
TRAINING_CONFIG = {
    "output_dir": "medgemma4b_it_grpo_reasoning",
    "per_device_train_batch_size": 2,  # Increased for A100 80GB
    "gradient_accumulation_steps": 4,  # Reduced since we increased batch size
    "learning_rate": 5e-6,
    "num_train_epochs": 1,
    "num_generations": 4,  # Number of completions generated per prompt
    "generation_batch_size": 4,  # Must be divisible by num_generations (4)
    "max_prompt_length": None,  # Don't truncate prompts (important for VLM)
    "max_completion_length": 128,
    "bf16": True,
    "remove_unused_columns": False,
    "logging_steps": 10,
    "save_steps": 500,
    "report_to": "none",
    "dataloader_num_workers": 0,  # Use 0 for iterable datasets to avoid multiprocessing issues
    "gradient_checkpointing": True,
    "warmup_steps": 50,
    "beta": 0.0,  # No KL penalty
    "scale_rewards": True,
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": "CAUSAL_LM"
}

# Quantization configuration
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16"
}

# Reward function weights
REWARD_WEIGHTS = {
    "correct_answer": 1.0,      # Weight for correct letter
    "explanation_quality": 0.6,  # Weight for explanation F1 score
    "evidence_usage": 0.2,       # Weight for using report evidence
    "format_bonus": 0.1,         # Bonus for correct format
    "conciseness_bonus": 0.1,    # Bonus for concise responses
}

# Response format tokens
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

# Image processing settings
IMAGE_CONFIG = {
    "target_size": (512, 512),  # Resize images to this size
    "convert_16bit": True,      # Convert 16-bit images to 8-bit
    "normalize_range": True,    # Normalize pixel values to 0-255
    "force_rgb": True,          # Convert all images to RGB
}

# Iterable dataset settings (following HuggingFace best practices)
ITERABLE_DATASET_CONFIG = {
    "use_iterable_datasets": True,  # Use IterableDataset for large datasets
    "streaming": True,              # Stream data without caching to disk
    "on_demand_loading": True,      # Load images only when needed
}

# Dataset processing
DATASET_SAVE_NAMES = {
    "train": "radiology_mcq_train",
    "val": "radiology_mcq_val", 
    "test": "radiology_mcq_test"
}