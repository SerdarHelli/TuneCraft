#!/usr/bin/env python3
"""
Script to help optimize cache size for lazy loading based on system memory
"""

from memory_utils import get_memory_info, estimate_cache_size, recommend_cache_size, print_memory_info
import json

def main():
    print("🔍 MedGemma Cache Size Optimizer")
    print("=" * 50)
    
    # Get current memory info
    print_memory_info("Current System Status")
    
    # Get memory estimates
    print("\n📊 Memory Usage Estimates")
    print("-" * 30)
    estimates = estimate_cache_size(image_size=(512, 512))
    print(f"Memory per image: {estimates['per_image_mb']} MB")
    print()
    
    print("Cache Size Options:")
    for cache_name, cache_info in estimates["estimates"].items():
        images = cache_info['images']
        memory_gb = cache_info['memory_gb']
        print(f"  {images:4d} images → {memory_gb:5.2f} GB RAM")
    
    # Get recommendation
    print(f"\n🎯 Recommendations")
    print("-" * 20)
    
    memory_info = get_memory_info()
    available_gb = memory_info["system"]["available_gb"]
    
    # Conservative recommendation (70% of available memory)
    conservative = recommend_cache_size(available_gb, safety_factor=0.7)
    conservative_gb = conservative * estimates['per_image_mb'] / 1024
    
    # Aggressive recommendation (90% of available memory)
    aggressive = recommend_cache_size(available_gb, safety_factor=0.9)
    aggressive_gb = aggressive * estimates['per_image_mb'] / 1024
    
    print(f"Available RAM: {available_gb:.1f} GB")
    print(f"Conservative (70%): {conservative:4d} images ({conservative_gb:.2f} GB)")
    print(f"Aggressive (90%):   {aggressive:4d} images ({aggressive_gb:.2f} GB)")
    
    # GPU memory check
    if memory_info["gpu"]:
        print(f"\n🚀 GPU Memory Status")
        print("-" * 20)
        for gpu_name, gpu_info in memory_info["gpu"].items():
            free_gb = gpu_info["free_gb"]
            print(f"{gpu_info['name']}: {free_gb:.1f} GB free")
            
            if free_gb < 12:
                print("⚠️  Warning: Low GPU memory. Consider reducing batch size.")
            elif free_gb > 20:
                print("✅ Excellent GPU memory for training!")
    
    # Generate config recommendation
    print(f"\n⚙️  Recommended Config")
    print("-" * 20)
    
    recommended_cache = conservative
    recommended_workers = min(4, max(1, int(available_gb / 8)))  # 1 worker per 8GB RAM
    
    config_recommendation = {
        "LAZY_LOADING_CONFIG": {
            "cache_size": recommended_cache,
            "use_lazy_loading": True,
            "num_workers": recommended_workers,
        },
        "TRAINING_CONFIG": {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "dataloader_num_workers": recommended_workers,
        }
    }
    
    print("Add this to your config.py:")
    print()
    for key, value in config_recommendation.items():
        print(f"{key} = {json.dumps(value, indent=4)}")
    
    # Performance tips
    print(f"\n💡 Performance Tips")
    print("-" * 20)
    print("1. Start with conservative cache size and monitor memory usage")
    print("2. Increase cache size if you have spare RAM during training")
    print("3. Use SSD storage for faster image loading")
    print("4. Monitor GPU memory usage and adjust batch size accordingly")
    
    if available_gb < 16:
        print("⚠️  Your system has limited RAM. Consider:")
        print("   - Using smaller cache size (500-1000)")
        print("   - Reducing num_workers to 1-2")
        print("   - Closing other applications during training")
    
    print(f"\n🎉 Optimization complete!")
    print(f"Recommended cache size: {recommended_cache} images")

if __name__ == "__main__":
    main()