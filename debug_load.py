import torch
from diffusers import AutoPipelineForText2Image
import traceback

try:
    model_id = "segmind/SSD-1B"
    print(f"[*] Testing loading {model_id} with AutoPipeline...")
    
    # Check if we can just load the components manually to save memory
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer
    
    print("[*] Stage 1: Loading Tokenizer")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    print("[+] Tokenizer loaded.")
    
    print("[*] Stage 2: Loading Text Encoder")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    print("[+] Text Encoder loaded.")

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, 
        torch_dtype=torch.float32
    )
    print("[+] Successfully loaded with AutoPipeline.")
    print(f"Pipeline class: {pipe.__class__.__name__}")
except Exception as e:
    print("[!] Error at some stage:")
    traceback.print_exc()
