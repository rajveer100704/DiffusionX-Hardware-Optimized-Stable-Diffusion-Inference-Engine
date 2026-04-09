import torch
import numpy as np
from tqdm.auto import tqdm

class DiffusionXEngine:
    """
    Custom inference engine for Stable Diffusion with interceptable sampling loop.
    """
    def __init__(self, components, compile_model=False):
        self.vae = components["vae"]
        self.unet = components["unet"]
        self.tokenizer = components["tokenizer"]
        self.text_encoder = components["text_encoder"]
        self.scheduler = components["scheduler"]
        self.device = components["device"]
        self.dtype = components.get("dtype", torch.float32)
        
        # 0. Apply optimizations
        if compile_model:
            print("[*] Compiling UNet with torch.compile...")
            try:
                # Use reduce-overhead for inference speed
                self.unet = torch.compile(self.unet, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[!] torch.compile failed (falling back): {e}")

    def set_attention_processor(self, processor_type="sdpa"):
        """
        Switches between different attention backends.
        """
        print(f"[*] Setting attention processor to: {processor_type}")
        if processor_type == "xformers":
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception:
                print("[!] xformers error, falling back to SDPA")
        elif processor_type == "sdpa":
            # SDPA is default in Torch 2.0+
            pass 
        
    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=50, scheduler=None, batch_size=1, guidance_scale=7.5, seed=None, callback=None):
        if scheduler:
            self.scheduler = scheduler
            
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
        # 1. Encode prompt
        text_inputs = self.tokenizer(
            [prompt] * batch_size, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # 2. Get unconditional embeddings for classifier-free guidance
        uncond_input = self.tokenizer(
            [""] * batch_size, 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        
        # Concatenate for single pass
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # 3. Prepare latents
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, 64, 64),
            generator=generator,
            device=self.device
        )
        
        # 4. Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # 5. Denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="[*] Denoising")):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Callback for visualization progression
            if callback:
                callback(i, t, latents)
                
        # 6. Decode latents
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        
        # Post-process
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        
        return image
