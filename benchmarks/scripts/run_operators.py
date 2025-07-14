import torch
import numpy as np
import time
import os
import sys
import torch_npu

def do_bench(fn, num_iter=100, num_warmup=100):
    """Benchmark a function with warmup iterations"""
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    times = np.zeros(num_iter + num_warmup)

    for i in range(num_iter + num_warmup):
        with torch.no_grad():
            start.record()
            fn()
            end.record()
        torch.npu.synchronize()
        times[i] = start.elapsed_time(end)

    times = times[num_warmup:]
    elapsed_time = np.amin(times) / 1000  # Convert to seconds
    return elapsed_time

def calculate_performance(flops, elapsed_time, memory_bytes=None):
    """Calculate performance metrics and automatically choose appropriate unit"""
    # 计算FLOPS性能
    if flops / elapsed_time >= 1e12:
        performance = flops / elapsed_time / 1e12
        unit = "TFLOPS"
    elif flops / elapsed_time >= 1e9:
        performance = flops / elapsed_time / 1e9
        unit = "GFLOPS"
    elif flops / elapsed_time >= 1e6:
        performance = flops / elapsed_time / 1e6
        unit = "MFLOPS"
    else:
        performance = flops / elapsed_time / 1e3
        unit = "KFLOPS"
    
    # 计算带宽（如果提供了内存访问量）
    bandwidth_result = None
    if memory_bytes is not None:
        bandwidth_gbs = memory_bytes / elapsed_time / 1e9  # Convert to GB/s
        bandwidth_result = bandwidth_gbs
    
    return performance, unit, bandwidth_result

def print_benchmark_result(flops, elapsed_time, memory_bytes=None):
    """Print benchmark results with automatic unit selection"""
    performance, unit, bandwidth_result = calculate_performance(flops, elapsed_time, memory_bytes)
    print(f"{unit}: {performance:.2f}")
    print(f"Time: {elapsed_time*1000:.2f} ms")
    if bandwidth_result is not None:
        print(f"Bandwidth: {bandwidth_result:.2f} GB/s")
    return performance

def benchmark_matmul(m, k, n, dtype=torch.float16):
    """Benchmark matrix multiplication"""
    print(f"\n=== Matrix Multiplication Benchmark ===")
    print(f"Input shapes: ({m}, {k}) x ({k}, {n})")

    x = torch.randn(m, k, dtype=dtype, device="npu")
    w = torch.randn(k, n, dtype=dtype, device="npu")

    def matmul_fn():
        return torch.matmul(x, w)

    sec = do_bench(matmul_fn)
    flops = 2 * m * k * n

    bytes_per_element = 2 if dtype == torch.float16 else 4
    total_bytes = (m * k + k * n + m * n) * bytes_per_element
    
    return print_benchmark_result(flops, sec, total_bytes)

def benchmark_flash_attention(batch_size, seq_len, hidden_dim, num_heads, num_kv_heads=None, dtype=torch.float16):
    """Benchmark torch_npu._npu_flash_attention"""
    if num_kv_heads is None:
        num_kv_heads = num_heads  # Default to MHA if not specified
    head_dim = hidden_dim // num_heads
    
    print(f"\n=== Flash Attention Benchmark ===")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, Head dim: {head_dim}, Heads: {num_heads}, KV Heads: {num_kv_heads}")
    
    # Create input tensors
    num_tokens = batch_size * seq_len
    query = torch.randn(num_tokens, hidden_dim, dtype=dtype, device="npu")
    key = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=dtype, device="npu")
    value = torch.randn(num_tokens, num_kv_heads * head_dim, dtype=dtype, device="npu")
    
    # Create causal attention mask - match the format from attention.py
    mask = torch.ones(seq_len, seq_len, dtype=dtype, device="npu")
    mask = mask.tril()  # Lower triangular mask
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, seq_len, seq_len)
    
    def attention_fn():
        # Reshape tensors to BSH format
        query_reshaped = query.view(-1, num_heads, head_dim).contiguous()
        key_reshaped = key.view(-1, num_kv_heads, head_dim).contiguous()
        value_reshaped = value.view(-1, num_kv_heads, head_dim).contiguous()
        
        # Create output tensor
        output = torch.empty(num_tokens, num_heads, head_dim, dtype=dtype, device="npu")
        
        # Create seq_lens tensor
        seq_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device="cpu")
        
        # Call torch_npu._npu_flash_attention
        
        torch_npu._npu_flash_attention(
            query=query_reshaped,
            key=key_reshaped,
            value=value_reshaped,
            mask=mask,
            seq_len=seq_lens,
            scale_value=1.0 / (head_dim ** 0.5),
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            out=output
        )
        
        return output.view(num_tokens, hidden_dim)
    
    sec = do_bench(attention_fn)
    flops = 2 * batch_size * seq_len * seq_len * hidden_dim
    
    # 计算内存访问量 - Flash Attention
    bytes_per_element = 2 if dtype == torch.float16 else 4
    head_dim = hidden_dim // num_heads
    
    # Flash Attention内存访存：读取Q/K/V + 写入输出
    # Q: batch_size * seq_len * hidden_dim
    # K: batch_size * seq_len * (num_kv_heads * head_dim)  
    # V: batch_size * seq_len * (num_kv_heads * head_dim)
    # Output: batch_size * seq_len * hidden_dim
    total_bytes = (batch_size * seq_len * (hidden_dim + 2 * num_kv_heads * head_dim + hidden_dim)) * bytes_per_element
    
    return print_benchmark_result(flops, sec, total_bytes)

def benchmark_paged_attention(batch_size, seq_len, num_heads, num_kv_heads, head_dim, dtype=torch.float16):
    """完全对齐 attention.py decode 阶段的 paged_attention 单算子流程"""
    hidden_dim = num_heads * head_dim
    block_size = 64
    num_blocks = 501

    print(f"\n=== Paged Attention Benchmark ===")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, Head dim: {head_dim}, Heads: {num_heads}, KV Heads: {num_kv_heads}")

    # 1. 生成 key_cache/value_cache
    kv_cache_shape = (num_blocks, block_size, num_kv_heads * head_dim // 16, 16)
    key_cache = torch.randn(kv_cache_shape, dtype=dtype, device="npu")
    value_cache = torch.randn(kv_cache_shape, dtype=dtype, device="npu")
    key_cache = torch_npu.npu_format_cast(key_cache, 29)
    value_cache = torch_npu.npu_format_cast(value_cache, 29)

    # 2. 生成 block_table/context_lens
    # block_table是二维，第一维度是batch_size，第二维度是seq_len/block_size向上取整，数值是0~num_blocks-1
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size  # 向上取整
    # 生成有规律的block_table，每个序列的block是连续的
    block_table = torch.zeros((batch_size, num_blocks_per_seq), dtype=torch.int32, device="npu")
    for i in range(batch_size):
        for j in range(num_blocks_per_seq):
            block_table[i, j] = i * num_blocks_per_seq + j
    context_lens = torch.tensor([seq_len] * batch_size, dtype=torch.int32, device="npu")

    # 3. 生成 query/output
    query = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="npu")
    output = torch.empty(batch_size, num_heads, head_dim, dtype=dtype, device="npu")

    def paged_attention_fn():
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            scale_value=1.0 / (head_dim ** 0.5),
            block_table=block_table,
            context_lens=context_lens,
            out=output
        )
        return output

    # 5. 运行benchmark并计算性能
    sec = do_bench(paged_attention_fn)
    
    # 计算FLOPs: 对于paged attention decode阶段，主要是Q*K^T和attention*V的计算
    flops = 4 * batch_size * hidden_dim * seq_len
    
    # 计算内存访问量
    bytes_per_element = 2 if dtype == torch.float16 else 4
    # Paged Attention带宽：读取query + 读取KV cache + 写入输出
    total_bytes = (batch_size * num_heads * head_dim + 
                   batch_size * seq_len * num_kv_heads * head_dim * 2 +  # K和V
                   batch_size * num_heads * head_dim) * bytes_per_element
    
    return print_benchmark_result(flops, sec, total_bytes)

def benchmark_swiglu(batch_size, seq_len, hidden_dim, dtype=torch.float16):
    """Benchmark SwiGLU activation function"""
    print(f"\n=== SwiGLU Benchmark ===")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}, Hidden dim: {hidden_dim}")
    import vllm_ascend.vllm_ascend_C
    print(torch.ops._C._swiglu_fused)
    
    # SwiGLU input: typically comes from a linear layer, so we create a tensor
    # Note: SwiGLU expects input with last dimension = 2 * hidden_dim (gate + value)
    x = torch.randn(batch_size, seq_len, hidden_dim * 2, dtype=dtype, device="npu")
    
    def swiglu_fn():
        # Use NPU optimized SwiGLU operator
        return torch.ops._C._swiglu_fused(x)
    
    sec = do_bench(swiglu_fn)
    # Approximate FLOPs for SwiGLU: 3 * batch_size * seq_len * hidden_dim
    flops = 3 * batch_size * seq_len * hidden_dim
    
    # 计算内存访问量
    bytes_per_element = 2 if dtype == torch.float16 else 4
    # SwiGLU带宽：读取输入，写入输出
    # 输入: batch_size * seq_len * (hidden_dim * 2)
    # 输出: batch_size * seq_len * hidden_dim
    total_bytes = (batch_size * seq_len * (hidden_dim * 2 + hidden_dim)) * bytes_per_element
    
    return print_benchmark_result(flops, sec, total_bytes)

def benchmark_add_rms_norm(num_tokens, dim, dtype=torch.float16):
    """Benchmark Add + RMSNorm fused operator"""
    print(f"\n=== Add + RMSNorm Benchmark ===")
    print(f"Num tokens: {num_tokens}, Dim: {dim}")
    
    # Import the add_rms_norm operator
    import ascend910a_extras.ops as ops
    
    # Create input tensors
    x = torch.randn(num_tokens, dim, dtype=dtype, device="npu")
    residual = torch.randn(num_tokens, dim, dtype=dtype, device="npu")
    weight = torch.randn(dim, dtype=dtype, device="npu")
    epsilon = 1e-6
    
    def add_rms_norm_fn():
        # Use NPU optimized Add + RMSNorm operator
        return ops.add_rms_norm(x, residual, weight, epsilon)
    
    sec = do_bench(add_rms_norm_fn)
    
    # Calculate FLOPs for Add + RMSNorm:
    # 1. Add: batch_size * seq_len * hidden_dim additions
    # 2. Square: batch_size * seq_len * hidden_dim multiplications
    # 3. Sum reduction: batch_size * seq_len * hidden_dim additions
    # 4. Sqrt + epsilon: batch_size * seq_len operations
    # 5. Division: batch_size * seq_len * hidden_dim divisions
    # 6. Weight multiplication: batch_size * seq_len * hidden_dim multiplications
    flops = (num_tokens * dim +  # Add
             num_tokens * dim +  # Square
             num_tokens * dim +  # Sum reduction
             num_tokens +               # Sqrt + epsilon
             num_tokens * dim +  # Division
             num_tokens * dim)   # Weight multiplication
    flops = 6 * num_tokens * dim  # Simplified approximation
    
    # 计算内存访问量
    bytes_per_element = 2 if dtype == torch.float16 else 4
    # Add + RMSNorm带宽：
    # 读取: x + residual + weight
    # 写入: output
    total_bytes = (num_tokens * dim * 3 +  # x, residual, output
                   dim) * bytes_per_element  # weight
    
    return print_benchmark_result(flops, sec, total_bytes)

def main():
    """Main function to run all benchmarks"""
    print("Starting PyTorch NPU Operator Benchmarks")
    print("=" * 50)
    
    # Benchmark configurations
    # 模型config文件:https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
    # 模型定义:https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/model_executor/models/qwen3.py
    configs = {
        "matmul": {
            "m": 4096,
            "k": 4096, # hidden_size = 4096 
            "n": 512 # intermediate_size = 12288
        },
        "flash_attention": {
            "batch_size": 1,  # Real batch size from debug output
            "seq_len": 16384,      # Real seq_len from debug output
            "hidden_dim": 4096, # 32 heads * 128 head_dim
            "num_heads": 32,    # Real num_heads from debug output
            "num_kv_heads": 8   # Real num_kv_heads from debug output (GQA)
        },
        "paged_attention": {
            "batch_size": 1,
            "seq_len": 16384,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "dtype": torch.float16
        },
        "swiglu": {
            "batch_size": 1,
            "seq_len": 16384,
            "hidden_dim": 4096
        },
        "add_rms_norm": {
            "num_tokens": 16384,
            "dim": 4096
        }
    }
    
    results = {}
    
    # Run matmul benchmark
    results["matmul"] = benchmark_matmul(**configs["matmul"])
    
    # Run flash attention benchmark (prefill phase)
    results["flash_attention"] = benchmark_flash_attention(**configs["flash_attention"])
    
    # Run paged attention benchmark (decode phase)
    results["paged_attention"] = benchmark_paged_attention(**configs["paged_attention"])
    
    # Run swiglu benchmark
    results["swiglu"] = benchmark_swiglu(**configs["swiglu"])
    
    # Run add_rms_norm benchmark
    results["add_rms_norm"] = benchmark_add_rms_norm(**configs["add_rms_norm"])

if __name__ == "__main__":
    main()