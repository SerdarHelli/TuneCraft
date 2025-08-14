import torch
from unsloth import FastVisionModel
from transformers import TextStreamer
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from config import *
import json
from datasets import Dataset, IterableDataset
from PIL import Image
import os

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Load model and tokenizer using Unsloth
print(f"Loading model: {MODEL_ID}")

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=TRAINING_CONFIG["max_seq_length"],
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # Use 4bit quantization
)

# Add LoRA adapters
model = FastVisionModel.get_peft_model(
    model,
    r=LORA_CONFIG["r"],
    lora_alpha=LORA_CONFIG["lora_alpha"],
    lora_dropout=LORA_CONFIG["lora_dropout"],
    target_modules=LORA_CONFIG["target_modules"],
    use_gradient_checkpointing="unsloth",  # Use Unsloth's gradient checkpointing
    random_state=3407,
    use_rslora=False,  # Rank stabilized LoRA
    loftq_config=None,  # LoftQ
)

print("Model loaded successfully with LoRA adapters")

def create_iterable_dataset_generator(json_path, images_base_path="."):
    """Create a generator for iterable dataset"""
    def generator():
        print(f"Loading dataset from {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        processed = 0
        
        # Handle both dict and list formats
        if isinstance(data, dict):
            items = data.values()
        else:
            items = data
            
        for item in items:
            # Get image path - handle different JSON structures
            image_path = None
            if 'image_path' in item:
                raw_path = item['image_path']
                # Handle relative paths that start with ../
                if raw_path.startswith('../'):
                    # Convert ../path to absolute path from project root
                    # ../data/images/file.png -> data/images/file.png
                    clean_path = raw_path.replace('../', '/home/')
                    image_path = os.path.normpath(clean_path)
                else:
                    image_path = os.path.join(images_base_path, raw_path)
            elif 'ImagePath' in item and item['ImagePath']:
                # Handle list of image paths
                image_paths = item['ImagePath']
                if isinstance(image_paths, list) and image_paths:
                    raw_path = image_paths[0]  # Use first image
                else:
                    raw_path = image_paths
                
                # Handle relative paths that start with ../
                if raw_path.startswith('../'):
                    clean_path = raw_path.replace('../', '/home/')
                    image_path = os.path.normpath(clean_path)
                else:
                    image_path = raw_path
            
            # Skip if no image path or image doesn't exist
            if not image_path or not os.path.exists(image_path):
                continue
                
            # Create conversation format for SFT
            user_message = item.get('question', '')
            
            # Construct the expected response with reasoning and solution format
            reasoning = item.get('heur_reason', item.get('reason', ''))
            answer_letter = item.get('answer', item.get('correct_answer', ''))
            explanation = item.get('explanation', item.get('correct_answer_explanation', ''))
            
            # Format the assistant response
            assistant_response = f"{REASONING_START}\n{reasoning}\n{REASONING_END}\n\n{SOLUTION_START}\n{answer_letter}: {explanation}\n{SOLUTION_END}"
            
            conversation = [
                {
                    "from": "human",
                    "value": f"<image>\n{user_message}"
                },
                {
                    "from": "gpt", 
                    "value": assistant_response
                }
            ]
            
            yield {
                "image": [image_path],
                "conversations": conversation
            }
            
            processed += 1
            if processed % 1000 == 0:
                print(f"Processed {processed} samples...")
    
    return generator

def formatting_prompts_func(examples):
    """Format conversations for training"""
    images = examples["image"]
    conversations = examples["conversations"]
    texts = []
    
    for image_path, conversation in zip(images, conversations):
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
            
        # Format conversation
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    return {
        "text": texts,
        "image": [Image.open(img).convert('RGB') for img in images]
    }

# Create iterable datasets
print("Creating iterable training dataset...")
train_generator = create_iterable_dataset_generator(TRAIN_JSON)
train_dataset = IterableDataset.from_generator(train_generator)

# Create validation dataset if available
val_dataset = None
if os.path.exists(VAL_JSON):
    print("Creating iterable validation dataset...")
    val_generator = create_iterable_dataset_generator(VAL_JSON)
    val_dataset = IterableDataset.from_generator(val_generator)

print("Iterable datasets created successfully")
print("💾 Images will be loaded on-demand during training")

# Training arguments
training_args = TrainingArguments(
    output_dir=TRAINING_CONFIG["output_dir"],
    per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
    warmup_steps=TRAINING_CONFIG["warmup_steps"],
    num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
    learning_rate=TRAINING_CONFIG["learning_rate"],
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=TRAINING_CONFIG["logging_steps"],
    optim="adamw_8bit",
    weight_decay=TRAINING_CONFIG["weight_decay"],
    lr_scheduler_type=TRAINING_CONFIG["lr_scheduler_type"],
    seed=3407,
    save_steps=TRAINING_CONFIG["save_steps"],
    save_total_limit=TRAINING_CONFIG["save_total_limit"],
    dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],
    remove_unused_columns=TRAINING_CONFIG["remove_unused_columns"],
    report_to=TRAINING_CONFIG["report_to"],
)

# Create SFT trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    formatting_func=formatting_prompts_func,
    max_seq_length=TRAINING_CONFIG["max_seq_length"],
    dataset_num_proc=2,
    packing=False,  # Don't pack sequences for vision models
    args=training_args,
)

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Start training
print("Starting SFT training...")
trainer_stats = trainer.train()

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# Save model
print("Saving trained model...")
model.save_pretrained(TRAINING_CONFIG["output_dir"])
tokenizer.save_pretrained(TRAINING_CONFIG["output_dir"])

# Save to GGUF format for inference (optional)
print("Saving model in GGUF format...")
model.save_pretrained_gguf(
    f"{TRAINING_CONFIG['output_dir']}_gguf", 
    tokenizer, 
    quantization_method="q4_k_m"
)

print("Training completed!")