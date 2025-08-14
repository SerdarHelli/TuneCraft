"""
Memory monitoring utilities for large-scale training
"""

import psutil
import torch
import gc
from typing import Dict, Any

def get_memory_info() -> Dict[str, Any]:
    """Get current memory usage information"""
    # System memory
    system_memory = psutil.virtual_memory()
    
    # GPU memory (if available)
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            total = gpu_memory.total_memory / 1024**3            # GB
            
            gpu_info[f"gpu_{i}"] = {
                "name": gpu_memory.name,
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(total - reserved, 2),
                "utilization": round((reserved / total) * 100, 1)
            }
    
    return {
        "system": {
            "total_gb": round(system_memory.total / 1024**3, 2),
            "available_gb": round(system_memory.available / 1024**3, 2),
            "used_gb": round(system_memory.used / 1024**3, 2),
            "percent": system_memory.percent
        },
        "gpu": gpu_info
    }

def print_memory_info(prefix: str = ""):
    """Print current memory usage"""
    info = get_memory_info()
    
    if prefix:
        print(f"\n=== {prefix} ===")
    
    # System memory
    sys = info["system"]
    print(f"💾 System RAM: {sys['used_gb']:.1f}GB / {sys['total_gb']:.1f}GB ({sys['percent']:.1f}%)")
    
    # GPU memory
    for gpu_name, gpu in info["gpu"].items():
        print(f"🚀 {gpu['name']}: {gpu['reserved_gb']:.1f}GB / {gpu['total_gb']:.1f}GB ({gpu['utilization']:.1f}%)")

def cleanup_memory():
    """Clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def estimate_cache_size(image_size: tuple = (512, 512), channels: int = 3, dtype_bytes: int = 1) -> Dict[str, float]:
    """
    Estimate memory usage for image cache
    
    Args:
        image_size: (width, height) of images
        channels: Number of channels (3 for RGB)
        dtype_bytes: Bytes per pixel (1 for uint8, 4 for float32)
    
    Returns:
        Dictionary with memory estimates
    """
    pixels_per_image = image_size[0] * image_size[1] * channels
    bytes_per_image = pixels_per_image * dtype_bytes
    mb_per_image = bytes_per_image / 1024**2
    
    # Estimate for different cache sizes
    cache_sizes = [100, 500, 1000, 2000, 5000]
    estimates = {}
    
    for cache_size in cache_sizes:
        total_mb = cache_size * mb_per_image
        total_gb = total_mb / 1024
        estimates[f"cache_{cache_size}"] = {
            "images": cache_size,
            "memory_mb": round(total_mb, 1),
            "memory_gb": round(total_gb, 2)
        }
    
    return {
        "per_image_mb": round(mb_per_image, 2),
        "estimates": estimates
    }

def recommend_cache_size(available_gb: float = None, safety_factor: float = 0.7) -> int:
    """
    Recommend cache size based on available memory
    
    Args:
        available_gb: Available system memory in GB (auto-detect if None)
        safety_factor: Use only this fraction of available memory
    
    Returns:
        Recommended cache size
    """
    if available_gb is None:
        info = get_memory_info()
        available_gb = info["system"]["available_gb"]
    
    # Use safety factor
    usable_gb = available_gb * safety_factor
    
    # Estimate memory per image (512x512 RGB = ~0.75MB)
    mb_per_image = 0.75
    usable_mb = usable_gb * 1024
    
    recommended_cache = int(usable_mb / mb_per_image)
    
    # Clamp to reasonable range
    recommended_cache = max(100, min(recommended_cache, 10000))
    
    return recommended_cache

if __name__ == "__main__":
    print("=== Memory Analysis ===")
    print_memory_info("Current Memory Usage")
    
    print("\n=== Cache Size Estimates ===")
    estimates = estimate_cache_size()
    print(f"Memory per image: {estimates['per_image_mb']} MB")
    
    for cache_name, cache_info in estimates["estimates"].items():
        print(f"{cache_info['images']:4d} images: {cache_info['memory_gb']:5.2f} GB")
    
    print(f"\n=== Recommendation ===")
    recommended = recommend_cache_size()
    print(f"Recommended cache size: {recommended} images")
    
    recommended_gb = recommended * estimates['per_image_mb'] / 1024
    print(f"Estimated memory usage: {recommended_gb:.2f} GB")