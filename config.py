"""
Configuration file for MedGemma SFT training
"""

# Dataset paths - Windows paths
TRAIN_JSON = "/home/QA_json/train_vqa_data.json"
VAL_JSON = "/home/QA_json/valid_vqa_data.json"
TEST_JSON = "/home/QA_json/test_vqa_data.json"

# Model configuration
MODEL_ID = "google/medgemma-4b-it"  # Medical vision-language model
USE_ONLY_FIRST_IMAGE = False  # Set to True to use only the first image per sample

# Training configuration - SFT with Unsloth
TRAINING_CONFIG = {
    "output_dir": "medgemma4b_it_sft_reasoning",
    "per_device_train_batch_size": 2,  # Batch size
    "gradient_accumulation_steps": 4,  # Effective batch size = 2 * 4 = 8
    "learning_rate": 2e-4,  # Higher learning rate for SFT
    "num_train_epochs": 3,  # More epochs for SFT
    "max_steps": -1,  # Use epochs instead of max_steps
    "max_seq_length": 2048,  # Maximum sequence length
    "bf16": True,
    "remove_unused_columns": False,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 3,
    "report_to": "none",
    "dataloader_num_workers": 0,
    "gradient_checkpointing": True,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
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

# Response format tokens
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"