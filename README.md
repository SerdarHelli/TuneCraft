# MedGemma GRPO Fine-tuning

This repository contains code for fine-tuning Google's MedGemma-4B-IT model using Group Relative Policy Optimization (GRPO) for medical visual question answering tasks.

## Overview

The training pipeline uses:
- **Model**: Google MedGemma-4B-IT (Vision-Language Model)
- **Method**: GRPO (Group Relative Policy Optimization) 
- **Task**: Medical radiology multiple-choice questions with chest X-ray images
- **Framework**: Hugging Face TRL (Transformer Reinforcement Learning)

## Key Features

- ✅ **Correct GRPO Implementation**: Uses latest TRL GRPO trainer with proper VLM support
- ✅ **Prompt-Only Dataset Format**: Properly formatted for GRPO training
- ✅ **Custom Reward Function**: Rewards correct answers, reasoning quality, and format
- ✅ **Memory Efficient**: Uses 4-bit quantization and LoRA for efficient training
- ✅ **VLM Support**: Handles images and text inputs correctly
- ✅ **Lazy Image Loading**: Handles 600k+ images without loading all into memory
- ✅ **16-bit Image Support**: Automatically converts 16-bit medical images to 8-bit RGB
- ✅ **Smart Caching**: LRU cache for frequently accessed images

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Update dataset paths in `config.py`:
```python
# Change these to your actual paths
TRAIN_JSON = "/path/to/your/train_vqa_data.json"
VAL_JSON   = "/path/to/your/valid_vqa_data.json"
TEST_JSON  = "/path/to/your/test_vqa_data.json"
```

3. Configure lazy loading for your system:
```python
# In config.py, adjust cache size based on your RAM
LAZY_LOADING_CONFIG = {
    "cache_size": 1000,  # Adjust based on available RAM
    "use_lazy_loading": True,
    "num_workers": 4,    # Parallel image loading
}
```

## Usage

### Quick Start
```bash
python run_training.py
```

### Step by Step

1. **Prepare datasets**:
```bash
python dataset_pre.py
```

2. **Run GRPO training**:
```bash
python train.py
```

## Dataset Format

The code expects JSON files with this structure:
```json
{
  "sample_id": {
    "question": "What is the primary finding?",
    "options": ["A. Normal", "B. Pneumonia", "C. Fracture", "D. Tumor"],
    "correct_answer": "B",
    "correct_answer_explanation": "Consolidation visible in right lower lobe",
    "ImagePath": ["/path/to/image.jpg"],
    "Indication": "Chest pain",
    "Findings": "Right lower lobe consolidation",
    "Impression": "Pneumonia"
  }
}
```

## GRPO Training Details

### Dataset Format
- **Type**: Prompt-only (required for GRPO)
- **Columns**: `prompt`, `image`, plus reward function columns
- **Images**: Single PIL Image per sample

### Reward Function
The custom reward function evaluates:
1. **Correct Answer** (1.0 points): Letter matches gold standard
2. **Explanation Quality** (0.6 points): Token F1 with gold explanation  
3. **Evidence Usage** (0.2 points): Uses words from radiology report
4. **Format Compliance** (0.1 points): Follows required format
5. **Conciseness** (0.1 points): Response ≤ 220 characters

### Model Configuration
- **Base Model**: google/medgemma-4b-it
- **Quantization**: 4-bit with NF4
- **LoRA**: r=16, α=32, dropout=0.05
- **Target Modules**: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj

### Training Configuration
- **Batch Size**: 1 per device, 8 gradient accumulation steps
- **Learning Rate**: 5e-6
- **Generations**: 4 per prompt
- **Max Completion**: 128 tokens
- **No KL Penalty**: β=0.0 (following recent best practices)

## Expected Output Format

The model is trained to respond in this format:
```
<start_working_out>
[Reasoning based on image and clinical context]
<end_working_out>
<SOLUTION>
B - Pneumonia visible in right lower lobe
</SOLUTION>
```

## Hardware Requirements

### For 600k Image Dataset:
- **GPU**: 16GB+ VRAM recommended (RTX 4090, A100, etc.)
- **RAM**: 32GB+ system RAM (with lazy loading, only ~2-4GB used for image cache)
- **Storage**: 100GB+ free space for model and datasets

### Memory Usage with Lazy Loading:
- **Without Lazy Loading**: 600k × 512×512×3 = ~450GB RAM (impossible!)
- **With Lazy Loading**: Only cache size × 0.75MB = ~750MB-2GB RAM ✅

### Cache Size Recommendations:
- **16GB RAM**: cache_size = 1000 (~750MB for images)
- **32GB RAM**: cache_size = 2000 (~1.5GB for images)  
- **64GB RAM**: cache_size = 4000 (~3GB for images)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `per_device_train_batch_size` to 1
   - Increase `gradient_accumulation_steps`
   - Enable `gradient_checkpointing=True`

2. **Image Loading Errors**:
   - Check image paths in JSON files
   - Ensure images are readable PIL format
   - Code includes fallback to placeholder images

3. **Dataset Format Errors**:
   - Ensure JSON structure matches expected format
   - Check that all required fields are present
   - Verify image paths are absolute or relative to script location

### Performance Tips

1. **Use vLLM for faster generation** (if available):
```python
training_args = GRPOConfig(
    # ... other args
    use_vllm=True,
    vllm_mode="colocate",
)
```

2. **Adjust generation settings**:
   - Reduce `num_generations` if memory constrained
   - Increase `max_completion_length` for longer reasoning

## References

- [GRPO Paper](https://arxiv.org/abs/2402.03300): DeepSeekMath: Pushing the Limits of Mathematical Reasoning
- [TRL GRPO Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [MedGemma Model](https://huggingface.co/google/medgemma-4b-it)

## License

This code is provided for research purposes. Please check the licenses of the underlying models and datasets.