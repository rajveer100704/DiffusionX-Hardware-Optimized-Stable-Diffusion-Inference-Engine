import torch
from diffusers import (
    StableDiffusionPipeline, 
    DDIMScheduler, 
    EulerDiscreteScheduler, 
    PNDMScheduler
)

class DiffusionModelLoader:
    """
    Handles loading of Stable Diffusion components and scheduler management.
    """
    def __init__(self, model_id: str, device: str = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing model {model_id} on {self.device}")
        
        # Determine data type
        self.dtype = torch.float32 if self.device == "cpu" else torch.float16
        
        # Load the pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=self.dtype,
            use_safetensors=True,
            safety_checker=None  # Disable for speed/memory on CPU
        ).to(self.device)
        
        # Extract components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.scheduler = self.pipe.scheduler
        
    def get_scheduler(self, name: str):
        """
        Returns a specific scheduler based on name.
        """
        config = self.pipe.scheduler.config
        if name.upper() == "DDIM":
            return DDIMScheduler.from_config(config)
        elif name.upper() == "EULER":
            return EulerDiscreteScheduler.from_config(config)
        elif name.upper() == "PNDM":
            return PNDMScheduler.from_config(config)
        else:
            print(f"[!] Scheduler {name} not found, defaulting to current.")
            return self.pipe.scheduler

    def get_components(self):
        return {
            "vae": self.vae,
            "unet": self.unet,
            "tokenizer": self.tokenizer,
            "text_encoder": self.text_encoder,
            "scheduler": self.scheduler,
            "device": self.device,
            "dtype": self.dtype
        }

    def enable_memory_optimizations(self, profile: str):
        """
        Applies memory-saving techniques based on the selected profile.
        """
        if profile == "low_vram":
            print("[*] Applying Low-VRAM optimizations (Sequential Offload + VAE Slicing)")
            try:
                self.pipe.enable_sequential_cpu_offload()
            except Exception as e:
                print(f"[!] Sequential offload failed: {e}")
            self.pipe.enable_vae_slicing()
            
        elif profile == "balanced":
            print("[*] Applying Balanced memory optimizations (Model Offload + VAE Slicing)")
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception as e:
                print(f"[!] Model offload failed: {e}")
            self.pipe.enable_vae_slicing()
            
        elif profile == "high_perf":
            print("[*] Running in High Performance mode (Minimal offloading)")
            self.pipe.enable_vae_slicing()
