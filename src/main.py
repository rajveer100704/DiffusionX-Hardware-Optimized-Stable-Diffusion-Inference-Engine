import os
import torch
import argparse
from omegaconf import OmegaConf

from models.stable_diffusion import DiffusionModelLoader
from pipeline.inference import DiffusionXEngine
from scheduler.custom_scheduler import DiffusionScheduler
from utils.visualization import Visualizer
from utils.reporting import ReportGenerator
from optimization.benchmarking import BenchmarkSuite
from optimization.memory import MemoryTracker
from optimization.hardware import HardwareDetector

def run_diffusion_x(args):
    # 1. Load configuration
    config = OmegaConf.load(args.config)
    
    # 2. Hardware Detection
    caps = HardwareDetector.get_capabilities()
    print("="*60)
    print(f"[*] DIFFUSIONX HARDWARE DETECTION")
    print(f"    - Device: {caps['device_type'].upper()}")
    print(f"    - VRAM: {caps['vram_gb']:.2f} GB")
    print(f"    - RAM: {caps['ram_gb']:.2f} GB")
    print("="*60)

    # 3. Model Loading
    model_id = config.model.prod_name if args.prod else config.model.dev_name
    loader = DiffusionModelLoader(model_id, device=caps["device_type"])
    
    # Apply Optimizations based on profile
    opt_profile = args.optimize or HardwareDetector.suggest_profile(caps)
    loader.enable_memory_optimizations(opt_profile)
    
    components = loader.get_components()
    
    if args.custom_scheduler:
        print("[*] Re-routing to explicitly managed CustomScheduler (DDPM)")
        components["scheduler"] = DiffusionScheduler(num_steps=args.steps, device=caps["device_type"])
        
    engine = DiffusionXEngine(components, compile_model=(opt_profile == "high_perf"))
    
    # 4. Handle Benchmarking Mode
    if args.benchmark:
        benchmark = BenchmarkSuite(config)
        reporter = ReportGenerator()
        
        # A. Systematic Matrix
        matrix_results = benchmark.run_systematic_matrix(
            engine, loader, 
            args.prompt, 
            config.inference.schedulers, 
            config.inference.steps_study, 
            config.inference.batch_sizes
        )
        reporter.generate_matrix_plots(matrix_results)
        
        # B. Optimization Impact Analysis
        # Create a baseline engine (no compile, defaults)
        baseline_engine = DiffusionXEngine(components, compile_model=False)
        impact = benchmark.compare_optimizations(baseline_engine, engine, loader, args.prompt)
        reporter.generate_impact_summary(impact)
        
        print(f"[*] Benchmarking and Reporting complete.")
    
    # 5. Standard Generation
    else:
        visualizer = Visualizer(config)
        print(f"[*] Generating: '{args.prompt}' [Batch: {args.batch}, Steps: {args.steps}]")
        
        images = engine.generate(
            args.prompt, 
            num_inference_steps=args.steps, 
            batch_size=args.batch,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        # Save results
        for i, img in enumerate(images):
            path = visualizer.save_image(img, f"output_{i}.png")
            print(f"[+] Saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffusionX: Hardware-Accelerated Multi-Backend Inference Engine")
    parser.add_argument("--prompt", type=str, default="A futuristic cyberpunk city, neon lights, 8k", help="Text prompt for generation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed for reproducibility")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG prompt adherence scaling")
    parser.add_argument("--optimize", type=str, choices=["high_perf", "low_vram", "balanced"], help="Optimization profile")
    parser.add_argument("--custom_scheduler", action="store_true", help="Use manual DDPM scheduler")
    parser.add_argument("--benchmark", action="store_true", help="Run full systematic benchmark suite")
    parser.add_argument("--prod", action="store_true", help="Use production model (SD v1.5)")
    
    args = parser.parse_args()
    run_diffusion_x(args)
