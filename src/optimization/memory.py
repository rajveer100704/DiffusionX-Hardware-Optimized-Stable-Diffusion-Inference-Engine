import torch
import psutil
import os

class MemoryTracker:
    """
    Tracks memory usage (RAM and VRAM).
    """
    @staticmethod
    def get_stats():
        stats = {}
        
        # System RAM
        vm = psutil.virtual_memory()
        stats["app_ram_usage_mb"] = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        stats["system_ram_percent"] = vm.percent
        
        # GPU VRAM
        if torch.cuda.is_available():
            stats["vram_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["vram_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            stats["vram_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            stats["vram_allocated_mb"] = 0
            stats["vram_reserved_mb"] = 0
            
        return stats

    @staticmethod
    def log_stats(tag=""):
        stats = MemoryTracker.get_stats()
        print(f"[*] Memory Statistics {tag}:")
        print(f"    - RAM Usage: {stats['app_ram_usage_mb']:.2f} MB")
        if torch.cuda.is_available():
            print(f"    - VRAM Allocated: {stats['vram_allocated_mb']:.2f} MB")
        return stats
