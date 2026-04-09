import pytest
import torch
from optimization.hardware import HardwareDetector

def test_hardware_detection():
    caps = HardwareDetector.get_capabilities()
    assert "device_type" in caps
    assert caps["device_type"] in ["cpu", "cuda", "mps"]
    
def test_profile_suggestion():
    caps = {"device_type": "cpu", "vram_gb": 0, "ram_gb": 16}
    profile = HardwareDetector.suggest_profile(caps)
    assert profile == "low_vram"
