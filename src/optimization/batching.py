import time

class BatchingOptimizer:
    """
    Analyzes throughput vs batch size.
    """
    def __init__(self, config):
        self.batch_sizes = config.inference.batch_sizes
        
    def run_throughput_study(self, engine, benchmark_suite, prompt):
        print(f"[*] Running Throughput Study: {self.batch_sizes}")
        study_results = []
        for bs in self.batch_sizes:
            _, latency = benchmark_suite.measure_latency(
                engine.generate, 
                prompt, 
                num_inference_steps=20, # Use low steps for faster benchmarking
                batch_size=bs
            )
            throughput = bs / latency
            study_results.append({
                "batch_size": bs, 
                "latency": latency, 
                "throughput": throughput
            })
            print(f"    - BS: {bs}, Latency: {latency:.2f}s, Throughput: {throughput:.2f} img/s")
            
        benchmark_suite.save_benchmark("throughput_study.json", study_results)
        return study_results
