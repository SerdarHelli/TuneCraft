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
python train.py
```

### 4. Test Dataset (Optional)

```bash
# Test iterable dataset functionality
python test_dataset.py
```

### 5. Test the Trained Model

```bash
# Test with validation data
python inference.py

# Interactive testing
python inference.py --interactive
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

### Model Output Format
The model learns to generate responses in this format:
```
<start_working_out>
[Medical reasoning based on image and context]
<end_working_out>

<SOLUTION>
A - [Detailed answer explanation]
</SOLUTION>
```

## Memory Requirements

- **Minimum:** 16GB VRAM (RTX 4080, RTX 3090)
- **Recommended:** 24GB VRAM (RTX 4090, RTX A6000)
- **Optimal:** 48GB+ VRAM (RTX 6000 Ada, A100)

Adjust batch size in `config.py` based on your GPU memory.

## Dataset Structure

```
data/
├── QA_json/
│   ├── train_vqa_data.json
│   ├── valid_vqa_data.json
│   └── test_vqa_data.json
└── images/
    └── [extracted medical images]
```

## Training Process

1. **Data Loading:** Iterable datasets from JSON files (memory efficient)
2. **Model Setup:** MedGemma-4B with Unsloth optimizations
3. **SFT Training:** Supervised learning on formatted conversations
4. **Model Saving:** Both PyTorch and GGUF formats

## Key Features

- **Iterable Datasets:** Images loaded on-demand, no memory limits
- **Memory Efficient:** Handle datasets of any size (600k+ images)
- **Fast Training:** 2x faster with Unsloth optimizations
- **Consumer GPU Ready:** Works on RTX 4090, RTX 3090, etc.

## Files

### Training
- `train.py` - Main SFT training script with iterable datasets
- `config.py` - Configuration for SFT training
- `requirements.txt` - Dependencies with Unsloth

### Data Processing
- `download_data.py` - Data download script

### Inference
- `inference.py` - Model inference script

### Setup & Testing
- `install_unsloth.py` - Dependency installation script
- `setup_and_train.py` - Complete setup and training pipeline
- `test_dataset.py` - Test iterable dataset functionality

## Advantages of Unsloth SFT

1. **Speed:** 2x faster training than standard methods
2. **Memory:** More efficient memory usage
3. **Compatibility:** Works with consumer GPUs
4. **Stability:** More stable than RL-based training
5. **Quality:** Better performance on supervised tasks

## License

This project is for research purposes. Please check the licenses of the underlying models and datasets.

## Troubleshooting

### CUDA Issues
```bash
# For CUDA 12.4 (your current setup)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1 (alternative)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues
- Reduce `per_device_train_batch_size` in `config.py`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Enable `gradient_checkpointing` (already enabled by default)

### Import Errors
```bash
# Run the installation script
python install_unsloth.py
```