#!/usr/bin/env python3
"""
Debug test script for SwiGLU function
"""

def swiglu_cpu(x):
    x1, x2 = x.chunk(2, -1)
    return (x1 * x1.sigmoid()) * x2

def debug_test():
    """Debug test to see detailed output and compare NPU/CPU results"""
    
    print("Debug SwiGLU test...")
    
    try:
        import torch
        import vllm_ascend.vllm_ascend_C
        
        print("Successfully imported modules")
        
        # Check if function is available
        if hasattr(torch.ops._C, '_swiglu'):
            print("_swiglu function is available")
        else:
            print("_swiglu function is not available")
            return False
        
        # Create small test tensor
        x_npu = torch.randn(1, 2, 4, device='npu', dtype=torch.float32)
        print(f"Input tensor: {x_npu.shape}, {x_npu.dtype}, {x_npu.device}")
        print(f"Input values: {x_npu[0, 0, :]}")
        
        # Call custom NPU function
        try:
            print("\nCalling _swiglu function...")
            y_npu = torch.ops._C._swiglu(x_npu)
            print(f"Function call successful: {x_npu.shape} -> {y_npu.shape}")
            print(f"Output values: {y_npu[0, 0, :]}")
            if torch.all(y_npu == 0):
                print("Output is all zeros - implementation may have issues")
            else:
                print("Output contains non-zero values")
        except Exception as e:
            print(f"Function call failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Compare with CPU result
        print("\nComparing with CPU PyTorch implementation...")
        x_cpu = x_npu.cpu()
        y_cpu = swiglu_cpu(x_cpu)
        y_npu_cpu = y_npu.cpu()
        print(f"CPU output: {y_cpu[0, 0, :]}")
        print(f"NPU output (to CPU): {y_npu_cpu[0, 0, :]}")
        diff = (y_cpu - y_npu_cpu).abs()
        print(f"Difference: {diff[0, 0, :]}")
        print(f"Max difference: {diff.max().item():.6e}")
        print(f"Mean difference: {diff.mean().item():.6e}")
        
        print("\nDebug test completed!")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_test()
    if success:
        print("\nDebug test completed!")
    else:
        print("\nDebug test failed.") 