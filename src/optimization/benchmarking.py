import time
import torch
import json
import os
import itertools
from optimization.memory import MemoryTracker

class BenchmarkSuite:
    """
    Automates research-grade latency & throughput measurements.
    """
    def __init__(self, config):
        self.output_dir = config.paths.benchmarks
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []

    def measure_performance(self, engine, prompt, steps, scheduler=None, batch_size=1):
        """
        Measures latency, throughput, and memory for a single run.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            
        start_mem = MemoryTracker.get_stats()
        start_time = time.time()
        
        # Run inference
        _ = engine.generate(
            prompt, 
            num_inference_steps=steps, 
            scheduler=scheduler, 
            batch_size=batch_size
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        end_mem = MemoryTracker.get_stats()
        
        latency = end_time - start_time
        throughput = (batch_size) / latency # images per second
        
        return {
            "latency": latency,
            "throughput": throughput,
            "mem_delta_mb": end_mem["app_ram_usage_mb"] - start_mem["app_ram_usage_mb"],
            "vram_peak_mb": end_mem.get("vram_max_allocated_mb", 0)
        }

    def run_systematic_matrix(self, engine, loader, prompt, schedulers, steps_list, batch_sizes):
        """
        Runs a full cross-product matrix (Scheduler x Steps x Batch).
        """
        print(f"[*] Starting Systematic Benchmark Matrix...")
        matrix_results = []
        
        combinations = list(itertools.product(schedulers, steps_list, batch_sizes))
        total = len(combinations)
        
        for i, (sched_name, steps, bs) in enumerate(combinations):
            print(f"    [{i+1}/{total}] Testing: {sched_name} | Steps: {steps} | Batch: {bs}")
            
            scheduler = loader.get_scheduler(sched_name)
            perf = self.measure_performance(engine, prompt, steps, scheduler, bs)
            
            matrix_results.append({
                "scheduler": sched_name,
                "steps": steps,
                "batch_size": bs,
                **perf
            })
            
        self.save_benchmark("systematic_matrix.json", matrix_results)
        return matrix_results

    def compare_optimizations(self, engine_baseline, engine_opt, loader, prompt):
        """
        Compares baseline vs optimized performance.
        """
        print("[*] Comparing Optimization Impact...")
        
        # Test baseline (30 steps, BS 1)
        baseline_perf = self.measure_performance(engine_baseline, prompt, 30, batch_size=1)
        
        # Test optimized (30 steps, BS 1)
        opt_perf = self.measure_performance(engine_opt, prompt, 30, batch_size=1)
        
        impact = {
            "baseline": baseline_perf,
            "optimized": opt_perf,
            "latency_improvement": (baseline_perf["latency"] - opt_perf["latency"]) / baseline_perf["latency"] * 100,
            "throughput_speedup": opt_perf["throughput"] / baseline_perf["throughput"]
        }
        
        self.save_benchmark("optimization_impact.json", impact)
        return impact

    def save_benchmark(self, filename, data):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"[+] Benchmark saved to {path}")
