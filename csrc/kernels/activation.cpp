#include "kernel_operator.h"
#include "kernel_tensor_impl.h"
#include "kernel_tpipe_impl.h"
#include "kernel_type.h"
#include <stdio.h>
#include <assert.h>
#include "types.h"
#include "utils.h"

__global__ __aicore__ void swiglu_kernel_f16(
    __gm__ uint8_t* input,
    __gm__ uint8_t* output,
    int dim, 
    int64_t stride,
    int64_t out_stride,
    int64_t num_tokens,
    int64_t block_dim
) {
    using scalar_t = half;
    using acc_t = float;
    __gm__ scalar_t *x_ptr = reinterpret_cast<__gm__ scalar_t *>(input);
    __gm__ scalar_t *y_ptr = reinterpret_cast<__gm__ scalar_t *>(output);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> in_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    AscendC::GlobalTensor<scalar_t> input_tensor;
    AscendC::GlobalTensor<scalar_t> output_tensor;

    // init
    pipe.InitBuffer(in_que, 1, sizeof(scalar_t) * dim * 2);
    pipe.InitBuffer(out_que, 1, sizeof(scalar_t) * dim);
    pipe.InitBuffer(calc_buf, 4 * dim * sizeof(acc_t));

    for (int64_t i = AscendC::GetBlockIdx(); i < num_tokens; i += block_dim) {
        input_tensor.SetGlobalBuffer(x_ptr + stride * i, dim * 2);
        output_tensor.SetGlobalBuffer(y_ptr + out_stride * i, dim);

        // global -> local
        AscendC::LocalTensor<scalar_t> x_local = in_que.AllocTensor<scalar_t>();
        AscendC::DataCopy(x_local, input_tensor, dim * 2);
        in_que.EnQue(x_local);

        AscendC::LocalTensor<scalar_t> x_local_deque = in_que.DeQue<scalar_t>(); 
        AscendC::LocalTensor<scalar_t> x_for_mul = x_local_deque;
        AscendC::LocalTensor<scalar_t> x_for_sigmoid = x_local_deque[dim];

        AscendC::LocalTensor<scalar_t> out_local = out_que.AllocTensor<scalar_t>();

        AscendC::LocalTensor<acc_t> x_for_mul_f32 = calc_buf.GetWithOffset<acc_t>(dim, 0);
        AscendC::LocalTensor<acc_t> x_for_sigmoid_f32 = calc_buf.GetWithOffset<acc_t>(dim, dim * sizeof(acc_t));
        AscendC::LocalTensor<acc_t> sigmoid_f32 = calc_buf.GetWithOffset<acc_t>(dim, 2 * dim * sizeof(acc_t));
        AscendC::LocalTensor<acc_t> prod_f32 = calc_buf.GetWithOffset<acc_t>(dim, 3 * dim * sizeof(acc_t));

        Cast(x_for_mul_f32, x_for_mul, AscendC::RoundMode::CAST_NONE, dim);
        Cast(x_for_sigmoid_f32, x_for_sigmoid, AscendC::RoundMode::CAST_NONE, dim);

        Sigmoid(sigmoid_f32, x_for_sigmoid_f32, dim);
        Mul(prod_f32, x_for_mul_f32, sigmoid_f32, dim);

        Cast(out_local, prod_f32, AscendC::RoundMode::CAST_TRUNC, dim);

        out_que.EnQue(out_local);

        AscendC::LocalTensor<scalar_t> out_local_deque = out_que.DeQue<scalar_t>();
        AscendC::DataCopy(output_tensor, out_local_deque, dim);
        out_que.FreeTensor(out_local_deque);
        in_que.FreeTensor(x_local_deque);
    }
}

namespace vllm_ascend {

void swiglu_impl(AscendType type, void *stream, uint8_t* input, uint8_t* output, int dim, int64_t stride, int64_t out_stride, int64_t num_tokens, uint32_t aiv_num) {
    int64_t block_dim = (num_tokens < 65535) ? num_tokens : 65535;
    if (type == AscendType::FP16) {
        swiglu_kernel_f16<<<block_dim, nullptr, stream>>>(input, output,
            dim, stride, out_stride, num_tokens, block_dim);
    } else {
        assert(false && "Unsupported data type for swiglu_impl");
    }
}
} // namespace vllm_ascend
