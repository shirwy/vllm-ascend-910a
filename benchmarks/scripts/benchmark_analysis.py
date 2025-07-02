#!/usr/bin/env python3
"""
Calculate FLOPS and Memory Bandwidth for Qwen3-8B model based on benchmark results.
This script can read benchmark results from JSON files in the benchmarks/results directory.
"""

import json
import os
import argparse
import glob
from typing import Dict, List, Tuple

# Qwen3-8B model configuration
QWEN3_8B_CONFIG = {
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 12288,
    "head_dim": 128,
    "vocab_size": 151936
}

# Model parameters (approximately 8B parameters)
MODEL_PARAMS = 8_000_000_000  # 8 billion parameters

# Data type size in bytes (bfloat16)
DTYPE_SIZE_BYTES = 2  # bfloat16 = 2 bytes

# Batch size used in benchmarks
BATCH_SIZE = 8

def load_benchmark_data_from_json(json_file_path: str) -> Dict:
    """
    Load benchmark data from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file
    
    Returns:
        Dictionary containing benchmark data
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract test name from filename
    filename = os.path.basename(json_file_path)
    test_name = filename.replace('.json', '')
    
    return {
        "test_name": test_name,
        "input_length": int(data["input_len"]),
        "output_length": int(data["output_len"]),
        "mean_latency_ms": data["avg_latency"] * 1000,  # Convert to milliseconds
        "model_name": data["model_name"]
    }

def load_all_benchmark_data(results_dir: str, pattern: str = "*.json") -> List[Dict]:
    """
    Load all benchmark data from JSON files in the results directory.
    
    Args:
        results_dir: Directory containing benchmark results
        pattern: File pattern to match (default: "*.json")
    
    Returns:
        List of benchmark data dictionaries
    """
    benchmark_data = []
    
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(results_dir, pattern))
    
    for json_file in json_files:
        try:
            data = load_benchmark_data_from_json(json_file)
            benchmark_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return benchmark_data

def calculate_attention_flops(input_length: int, output_length: int, config: Dict) -> int:
    """
    Calculate attention layer FLOPS for one layer.
    
    Args:
        input_length: Input sequence length
        output_length: Output sequence length  
        config: Model configuration
    
    Returns:
        FLOPS for attention layer
    """
    hidden_size = config["hidden_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    
    # Total sequence length (input + output)
    seq_len = input_length + output_length
    
    # QKV projection: hidden_size -> (num_heads + 2*num_kv_heads) * head_dim
    qkv_flops = hidden_size * (num_heads + 2 * num_kv_heads) * head_dim * seq_len * BATCH_SIZE
    
    # Attention computation: Q @ K^T
    # Q: (seq_len, num_heads, head_dim), K: (seq_len, num_kv_heads, head_dim)
    # Each head attends to its corresponding KV head (or shared KV heads)
    attention_flops = seq_len * seq_len * num_heads * head_dim * BATCH_SIZE
    
    # Attention output projection: (num_heads * head_dim) -> hidden_size
    output_proj_flops = num_heads * head_dim * hidden_size * seq_len * BATCH_SIZE
    
    return qkv_flops + attention_flops + output_proj_flops

def calculate_mlp_flops(input_length: int, output_length: int, config: Dict) -> int:
    """
    Calculate MLP layer FLOPS for one layer.
    
    Args:
        input_length: Input sequence length
        output_length: Output sequence length
        config: Model configuration
    
    Returns:
        FLOPS for MLP layer
    """
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    
    # Total sequence length (input + output)
    seq_len = input_length + output_length
    
    # Gate and Up projections: hidden_size -> intermediate_size
    gate_up_flops = 2 * hidden_size * intermediate_size * seq_len * BATCH_SIZE
    
    # Down projection: intermediate_size -> hidden_size
    down_flops = intermediate_size * hidden_size * seq_len * BATCH_SIZE
    
    return gate_up_flops + down_flops

def calculate_total_flops(input_length: int, output_length: int, config: Dict) -> int:
    """
    Calculate total FLOPS for the entire model.
    
    Args:
        input_length: Input sequence length
        output_length: Output sequence length
        config: Model configuration
    
    Returns:
        Total FLOPS
    """
    num_layers = config["num_hidden_layers"]
    
    # Calculate FLOPS for one layer
    attention_flops_per_layer = calculate_attention_flops(input_length, output_length, config)
    mlp_flops_per_layer = calculate_mlp_flops(input_length, output_length, config)
    
    # Total FLOPS for all layers
    total_flops = num_layers * (attention_flops_per_layer + mlp_flops_per_layer)
    
    return total_flops

def calculate_memory_bandwidth(input_length: int, output_length: int, config: Dict) -> int:
    """
    Calculate memory bandwidth usage in bytes.
    
    Args:
        input_length: Input sequence length
        output_length: Output sequence length
        config: Model configuration
    
    Returns:
        Memory bandwidth in bytes
    """
    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    
    # Total sequence length (input + output)
    seq_len = input_length + output_length
    
    # Input embeddings: vocab_size -> hidden_size
    input_embedding_memory = input_length * hidden_size * DTYPE_SIZE_BYTES * BATCH_SIZE
    
    # Output logits: hidden_size -> vocab_size
    output_logits_memory = output_length * vocab_size * DTYPE_SIZE_BYTES * BATCH_SIZE
    
    # Hidden states for each layer (simplified calculation)
    # Each layer processes the entire sequence
    num_layers = config["num_hidden_layers"]
    hidden_states_memory = num_layers * seq_len * hidden_size * DTYPE_SIZE_BYTES * BATCH_SIZE
    
    # Model parameters memory (8B parameters, read once per inference)
    model_params_memory = MODEL_PARAMS * DTYPE_SIZE_BYTES
    
    # Total memory bandwidth
    total_memory = model_params_memory + input_embedding_memory + output_logits_memory + hidden_states_memory
    
    return total_memory

def calculate_flops_per_second(flops: int, latency_ms: float) -> float:
    """
    Calculate FLOPS per second.
    
    Args:
        flops: Total FLOPS
        latency_ms: Latency in milliseconds
    
    Returns:
        FLOPS per second
    """
    latency_s = latency_ms / 1000.0
    return flops / latency_s

def calculate_bandwidth_per_second(memory_bytes: int, latency_ms: float) -> float:
    """
    Calculate memory bandwidth per second.
    
    Args:
        memory_bytes: Memory usage in bytes
        latency_ms: Latency in milliseconds
    
    Returns:
        Memory bandwidth per second in GB/s
    """
    latency_s = latency_ms / 1000.0
    bandwidth_gb_s = (memory_bytes / latency_s) / (1024**3)  # Convert to GB/s
    return bandwidth_gb_s

def format_flops(flops: float) -> str:
    """
    Format FLOPS in human readable format with fixed width.
    
    Args:
        flops: FLOPS value
    
    Returns:
        Formatted string with fixed width
    """
    if flops >= 1e12:
        return f"{flops/1e12:>6.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops/1e9:>6.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops/1e6:>6.2f} MFLOPS"
    else:
        return f"{flops:>6.2f} FLOPS"

def format_memory(memory_bytes: float) -> str:
    """
    Format memory in human readable format.
    
    Args:
        memory_bytes: Memory in bytes
    
    Returns:
        Formatted string
    """
    if memory_bytes >= 1e12:
        return f"{memory_bytes/1e12:.2f} TB"
    elif memory_bytes >= 1e9:
        return f"{memory_bytes/1e9:.2f} GB"
    elif memory_bytes >= 1e6:
        return f"{memory_bytes/1e6:.2f} MB"
    elif memory_bytes >= 1e3:
        return f"{memory_bytes/1e3:.2f} KB"
    else:
        return f"{memory_bytes:.2f} B"

def analyze_benchmarks(benchmark_data: List[Dict], config: Dict) -> List[Dict]:
    """
    Analyze benchmark data and calculate FLOPS and memory bandwidth.
    
    Args:
        benchmark_data: List of benchmark data dictionaries
        config: Model configuration
    
    Returns:
        List of analysis results
    """
    results = []
    
    for test_data in benchmark_data:
        input_length = test_data["input_length"]
        output_length = test_data["output_length"]
        latency_ms = test_data["mean_latency_ms"]
        
        # Calculate total FLOPS
        total_flops = calculate_total_flops(input_length, output_length, config)
        
        # Calculate FLOPS per second
        flops_per_sec = calculate_flops_per_second(total_flops, latency_ms)
        
        # Calculate memory bandwidth
        memory_bytes = calculate_memory_bandwidth(input_length, output_length, config)
        bandwidth_gb_s = calculate_bandwidth_per_second(memory_bytes, latency_ms)
        
        # Store results
        results.append({
            "test_name": test_data["test_name"],
            "input_length": input_length,
            "output_length": output_length,
            "total_tokens": input_length + output_length,
            "latency_ms": latency_ms,
            "total_flops": total_flops,
            "flops_per_sec": flops_per_sec,
            "memory_bytes": memory_bytes,
            "bandwidth_gb_s": bandwidth_gb_s,
            "model_name": test_data.get("model_name", "Unknown")
        })
    
    return results

def print_results(results: List[Dict], config: Dict):
    """
    Print analysis results in a concise format.
    
    Args:
        results: List of analysis results
        config: Model configuration
    """
    # Theoretical peak performance for 910A
    THEORETICAL_PEAK_TFLOPS = 221.270  # FP16
    THEORETICAL_PEAK_GBPS = 1182.66
    
    print("Qwen3-8B Benchmark Analysis Results")
    print("=" * 100)
    print(f"{'Test Name':<40} {'Input':<8} {'Output':<8} {'FLOPS':<15} {'Bandwidth':<8}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['test_name']:<40} {result['input_length']:<8} {result['output_length']:<8} "
              f"{format_flops(result['flops_per_sec']):<15} {result['bandwidth_gb_s']:<8.2f} GB/s")
    
    # Calculate and print summary
    avg_flops_per_sec = sum(r["flops_per_sec"] for r in results) / len(results)
    avg_bandwidth_gb_s = sum(r["bandwidth_gb_s"] for r in results) / len(results)
    
    print("-" * 100)
    print(f"Average: {format_flops(avg_flops_per_sec):<15} {avg_bandwidth_gb_s:<8.2f} GB/s")
    
    # Calculate utilization
    avg_flops_tflops = avg_flops_per_sec / 1e12
    flops_utilization = (avg_flops_tflops / THEORETICAL_PEAK_TFLOPS) * 100
    bandwidth_utilization = (avg_bandwidth_gb_s / THEORETICAL_PEAK_GBPS) * 100
    
    print(f"\nTheoretical Peak Performance (910A):")
    print(f"  Compute: {THEORETICAL_PEAK_TFLOPS} TFLOPS (FP16)")
    print(f"  Memory: {THEORETICAL_PEAK_GBPS} GB/s")
    print(f"\nAverage Utilization:")
    print(f"  Compute: {flops_utilization:.2f}%")
    print(f"  Memory: {bandwidth_utilization:.2f}%")

def main():
    """Main function to parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Analyze benchmark results and calculate FLOPS and memory bandwidth")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory containing benchmark results (default: results)")
    parser.add_argument("--pattern", default="*.json",
                       help="File pattern to match (default: *.json)")
    parser.add_argument("--model-config", type=str,
                       help="Path to model configuration JSON file (optional)")
    
    args = parser.parse_args()
    
    # Use default config or load from file
    config = QWEN3_8B_CONFIG
    if args.model_config and os.path.exists(args.model_config):
        with open(args.model_config, 'r') as f:
            config = json.load(f)
    
    # Load benchmark data
    print(f"Loading benchmark data from: {args.results_dir}")
    print(f"File pattern: {args.pattern}")
    print()
    
    benchmark_data = load_all_benchmark_data(args.results_dir, args.pattern)
    
    if not benchmark_data:
        print("No benchmark data found!")
        return
    
    print(f"Loaded {len(benchmark_data)} benchmark files")
    print()
    
    # Analyze benchmarks
    results = analyze_benchmarks(benchmark_data, config)
    
    # Print results
    print_results(results, config)

if __name__ == "__main__":
    main() 