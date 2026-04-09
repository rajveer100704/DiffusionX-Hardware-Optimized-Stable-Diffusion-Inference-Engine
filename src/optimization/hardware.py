import torch
import psutil
import platform

class HardwareDetector:
    """
    Detects hardware capabilities and suggests optimization profiles.
    """
    @staticmethod
    def get_capabilities():
        caps = {
            "device_type": "cpu",
            "vram_gb": 0,
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "os": platform.system(),
            "has_fp16_support": False,
            "has_bf16_support": False,
            "compute_capability": None
        }

        if torch.cuda.is_available():
            caps["device_type"] = "cuda"
            caps["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            caps["compute_capability"] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
            caps["has_fp16_support"] = True
            # BF16 support for Ampere+ (8.0+)
            caps["has_bf16_support"] = torch.cuda.get_device_capability(0)[0] >= 8
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            caps["device_type"] = "mps"
            caps["vram_gb"] = caps["ram_gb"] * 0.7  # Unified memory estimation
            # Apple Silicon supports FP16 well
            caps["has_fp16_support"] = True

        return caps

    @staticmethod
    def suggest_profile(caps):
        """
        Suggests an optimization profile based on hardware.
        """
        if caps["device_type"] == "cuda":
            if caps["vram_gb"] > 12:
                return "high_perf"
            elif caps["vram_gb"] < 6:
                return "low_vram"
            else:
                return "balanced"
        elif caps["device_type"] == "mps":
            return "balanced"
        else:
            return "low_vram" # CPU typically benefits from low_vram tactics to avoid RAM swap
