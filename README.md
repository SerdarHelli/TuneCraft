# MedGemma Fine-tuning with Unsloth SFT

This repository contains code for fine-tuning Google's MedGemma-4B model using Unsloth for efficient Supervised Fine-Tuning (SFT) on medical visual question answering tasks.

## Features

- **Ultra-fast training** with Unsloth (2x faster than standard fine-tuning)
- **Memory-efficient** with 4-bit quantization and LoRA
- **Vision-language support** for medical image analysis
- **Optimized for consumer GPUs** (RTX 4090, RTX 3090, etc.)
- **Medical image preprocessing** with automatic format conversion

## Quick Start

### 1. Install Dependencies

```bash
# Option 1: Use the install script
python install_unsloth.py

# Option 2: Manual installation
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -r requirements.txt
```

### 2. Download Data

```bash
python download_data.py
```

This will download the ReXVQA and ReXGradient datasets and organize them in the `data/` directory.

### 3. Start SFT Training

```bash
python train_sft.py
```

### 4. Test the Model

```bash
# Test with validation data
python inference_sft.py

# Interactive testing
python inference_sft.py --interactive
```

## Configuration

Edit `config.py` to customize:
- Dataset paths (now using Windows-compatible paths)
- Model settings
- Training hyperparameters
- LoRA configuration

## Key Changes from GRPO to SFT

### Training Approach
- **GRPO (old):** Reinforcement learning with custom reward function
- **SFT (new):** Supervised fine-tuning on question-answer pairs

### Performance Benefits
- **2x faster training** with Unsloth optimizations
- **Lower memory usage** (works on RTX 4090 24GB)
- **More stable training** (no RL complexity)
- **Better convergence** for supervised tasks

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

### Memory Usage with Iterable Datasets:
- **Without Iterable Datasets**: 600k × 512×512×3 = ~450GB RAM (impossible!)
- **With Iterable Datasets**: Images loaded on-demand = ~minimal RAM usage ✅

### Key Benefits:
- **No Memory Limits**: Handle datasets of any size (600k, 1M+ images)
- **No Disk Caching**: Datasets generated fresh each time (saves storage)
- **Automatic Cleanup**: Images garbage collected after use
- **Streaming**: Perfect for very large datasets that don't fit on disk

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