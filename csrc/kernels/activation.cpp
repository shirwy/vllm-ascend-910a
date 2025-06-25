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
    constexpr int BLOCK_SIZE_DIM = 128;
    __gm__ scalar_t *x_ptr = reinterpret_cast<__gm__ scalar_t *>(input);
    __gm__ scalar_t *y_ptr = reinterpret_cast<__gm__ scalar_t *>(output);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x0_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x1_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    AscendC::GlobalTensor<scalar_t> input_tensor;
    AscendC::GlobalTensor<scalar_t> output_tensor;

    // init
    pipe.InitBuffer(x0_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(x1_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(out_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(calc_buf, 5 * BLOCK_SIZE_DIM * sizeof(acc_t));

    for (int64_t i = AscendC::GetBlockIdx(); i < num_tokens; i += block_dim) {
        input_tensor.SetGlobalBuffer(x_ptr + stride * i, dim * 2);
        output_tensor.SetGlobalBuffer(y_ptr + out_stride * i, dim);

        // FIXME: no bound check
        for (int dim_i = 0; dim_i < dim; dim_i += BLOCK_SIZE_DIM) {
            AscendC::LocalTensor<scalar_t> x0_copy = x0_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> x1_copy = x1_que.AllocTensor<scalar_t>();
            AscendC::DataCopy(x0_copy, input_tensor[dim_i], BLOCK_SIZE_DIM);
            AscendC::DataCopy(x1_copy, input_tensor[dim + dim_i], BLOCK_SIZE_DIM);
            x0_que.EnQue(x0_copy);
            x1_que.EnQue(x1_copy);
            

            AscendC::LocalTensor<scalar_t> y = out_que.AllocTensor<scalar_t>();
            AscendC::LocalTensor<scalar_t> x0 = x0_que.DeQue<scalar_t>();
            AscendC::LocalTensor<scalar_t> x1 = x1_que.DeQue<scalar_t>();
            AscendC::LocalTensor<acc_t> x0_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 0);
            AscendC::LocalTensor<acc_t> x1_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, BLOCK_SIZE_DIM * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> sigmoid_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 2 * BLOCK_SIZE_DIM * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> prod_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 3 * BLOCK_SIZE_DIM * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> mul_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 4 * BLOCK_SIZE_DIM * sizeof(acc_t));
            Cast(x0_f32, x0, AscendC::RoundMode::CAST_NONE, BLOCK_SIZE_DIM);
            Cast(x1_f32, x1, AscendC::RoundMode::CAST_NONE, BLOCK_SIZE_DIM);
            Sigmoid(sigmoid_f32, x0_f32, BLOCK_SIZE_DIM);
            Mul(prod_f32, x0_f32, sigmoid_f32, BLOCK_SIZE_DIM);
            Mul(mul_f32, prod_f32, x1_f32, BLOCK_SIZE_DIM);
            Cast(y, mul_f32, AscendC::RoundMode::CAST_ODD, BLOCK_SIZE_DIM);
            out_que.EnQue(y);
            x0_que.FreeTensor(x0);
            x1_que.FreeTensor(x1);

            AscendC::LocalTensor<scalar_t> y_copy = out_que.DeQue<scalar_t>();
            AscendC::DataCopy(output_tensor[dim_i], y_copy, BLOCK_SIZE_DIM);
            out_que.FreeTensor(y_copy);
        }
    }
}

namespace vllm_ascend {

void swiglu_impl(AscendType type, void *stream, uint8_t* input, uint8_t* output, int dim, int64_t stride, int64_t out_stride, int64_t num_tokens, uint32_t aiv_num) {
    int64_t block_dim = (num_tokens < 65535) ? num_tokens : 65535;
    assert(dim >= 128 && dim % 128 == 0 && "swiglu_impl: dim must be a multiple of 128");
    if (type == AscendType::FP16) {
        swiglu_kernel_f16<<<block_dim, nullptr, stream>>>(input, output,
            dim, stride, out_stride, num_tokens, block_dim);
    } else {
        assert(false && "Unsupported data type for swiglu_impl");
    }
}
} // namespace vllm_ascend
