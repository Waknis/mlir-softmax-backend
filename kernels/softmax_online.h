#ifndef MLC_KERNELS_SOFTMAX_ONLINE_H
#define MLC_KERNELS_SOFTMAX_ONLINE_H

#include <cstdint>

// C ABI for the hand-written online softmax kernels. Each launch runs
// row-wise softmax: output[r, :] = softmax(input[r, :]) for r in [0, rows).
// Contiguous row-major layout, inner dimension is the reduction axis.
//
// Return value: 0 on success, non-zero cudaError_t on failure. The caller can
// pass the error code to mlc_softmax_online_last_error() to get a human
// string for logging.

#ifdef __cplusplus
extern "C" {
#endif

int mlc_softmax_online_f32(const float* d_input,
                           float* d_output,
                           int64_t rows,
                           int64_t cols,
                           void* cuda_stream);

int mlc_softmax_online_f16(const void* d_input,
                           void* d_output,
                           int64_t rows,
                           int64_t cols,
                           void* cuda_stream);

// Naive three-pass softmax (max -> exp+sum -> divide). Same ABI, useful as a
// "what does a straightforward CUDA implementation do" baseline.
int mlc_softmax_naive_f32(const float* d_input,
                          float* d_output,
                          int64_t rows,
                          int64_t cols,
                          void* cuda_stream);

const char* mlc_softmax_online_last_error(int error_code);

#ifdef __cplusplus
}
#endif

#endif  // MLC_KERNELS_SOFTMAX_ONLINE_H
