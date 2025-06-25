#!/usr/bin/env python3
"""
Debug test script for SwiGLU function
"""
import torch
import vllm_ascend.vllm_ascend_C

def swiglu_cpu(x):
    x1, x2 = x.chunk(2, -1)
    x1_f32, x2_f32 = x1.to(torch.float32), x2.to(torch.float32)
    out = (x1_f32 * x1_f32.sigmoid()) * x2_f32
    # out = x1_f32 * x2_f32
    # out = x1_f32
    return out.to(x.dtype)

if __name__ == "__main__":
    torch.manual_seed(0)
    num_tokens = 2048
    # dim = 12288
    dim = 1024
    dtype = torch.float16
    x_npu = torch.randn(num_tokens, dim * 2, device='npu', dtype=dtype)
    # x_npu = torch.arange(num_tokens * dim * 2).view(num_tokens, dim * 2).to(dtype).to('npu')
    print(f"Input tensor: {x_npu.shape}, {x_npu.dtype}, {x_npu.device}")
    print(f"{x_npu=}")
    x_cpu = x_npu.cpu()
    y_npu = torch.ops._C._swiglu_fused(x_npu)
    y_cpu = swiglu_cpu(x_cpu)
    torch.npu.synchronize()
    y_npu_cpu = y_npu.cpu()
    print(f"{y_cpu=}")
    print(f"{y_npu_cpu=}")
    torch.testing.assert_close(y_cpu, y_npu_cpu)
    print("PASS")
