// Row-wise softmax kernels: online (fused) and naive (three-pass).
//
// Algorithmic notes
// -----------------
// Online softmax (Milakov & Gimelshein 2018, FlashAttention 2022) tracks a
// running (m, d) pair where m is the max seen so far and d is the sum of
// exp(x_i - m) over elements seen so far. The merge rule for combining two
// partial results (m_a, d_a) and (m_b, d_b) is:
//     m_new = max(m_a, m_b)
//     d_new = d_a * exp(m_a - m_new) + d_b * exp(m_b - m_new)
// which is numerically equivalent to the safe 3-pass softmax but fuses the
// max and sum passes into a single traversal.
//
// Reduction strategy: thread-local strided scan -> warp-level reduction via
// __shfl_xor_sync -> block-level reduction via shared memory. One block per
// row. This is the standard GPU reduction pattern; no novel blocking here,
// the interesting piece is the (m, d) merge under warp shuffles.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

#include "softmax_online.h"

namespace {

constexpr int kWarpSize = 32;

// Identity for running max. `-FLT_MAX` instead of `-INFINITY` so the online
// merge math stays NaN-free under `--use_fast_math`: `__expf(-inf)` is not
// required to return 0 and in practice can return NaN, which taints a
// reduction if any lane carries the identity value. `-FLT_MAX` keeps every
// intermediate finite; `__expf(-FLT_MAX)` underflows cleanly to 0.
constexpr float kNegInfIdentity = -FLT_MAX;

// Merge two online partials that arrive via warp shuffle. Operates in-place
// on the caller's (m, d) pair. WARP_SIZE must be a power of two.
__device__ __forceinline__ void warpMergeOnline(float& m, float& d) {
  constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    float mPartner = __shfl_xor_sync(kMask, m, offset);
    float dPartner = __shfl_xor_sync(kMask, d, offset);
    float mNew = fmaxf(m, mPartner);
    d = d * __expf(m - mNew) + dPartner * __expf(mPartner - mNew);
    m = mNew;
  }
}

__device__ __forceinline__ float warpReduceMax(float v) {
  constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    v = fmaxf(v, __shfl_xor_sync(kMask, v, offset));
  }
  return v;
}

__device__ __forceinline__ float warpReduceSum(float v) {
  constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    v += __shfl_xor_sync(kMask, v, offset);
  }
  return v;
}

// ---------- Online (fused) softmax, f32 ----------
template <int BLOCK_SIZE>
__global__ void softmaxOnlineF32Kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int64_t cols) {
  constexpr int kNumWarps = BLOCK_SIZE / kWarpSize;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warpId = tid / kWarpSize;
  const int laneId = tid & (kWarpSize - 1);

  const float* rowIn = x + row * cols;
  float* rowOut = y + row * cols;

  // Thread-local online scan over a strided slice of the row.
  float m = kNegInfIdentity;
  float d = 0.f;
  for (int64_t i = tid; i < cols; i += BLOCK_SIZE) {
    float xi = rowIn[i];
    float mNew = fmaxf(m, xi);
    d = d * __expf(m - mNew) + __expf(xi - mNew);
    m = mNew;
  }

  // Warp-level merge.
  warpMergeOnline(m, d);

  // Inter-warp merge via shared memory. Max kNumWarps = 32 (for BLOCK_SIZE
  // up to 1024).
  __shared__ float sMax[kWarpSize];
  __shared__ float sSum[kWarpSize];
  if (laneId == 0) {
    sMax[warpId] = m;
    sSum[warpId] = d;
  }
  __syncthreads();

  if (warpId == 0) {
    m = (laneId < kNumWarps) ? sMax[laneId] : kNegInfIdentity;
    d = (laneId < kNumWarps) ? sSum[laneId] : 0.f;
    warpMergeOnline(m, d);
    if (laneId == 0) {
      sMax[0] = m;
      sSum[0] = d;
    }
  }
  __syncthreads();

  const float rowMax = sMax[0];
  const float invSum = 1.f / sSum[0];

  // Normalize pass. exp(xi - max) * (1 / sum).
  for (int64_t i = tid; i < cols; i += BLOCK_SIZE) {
    float xi = rowIn[i];
    rowOut[i] = __expf(xi - rowMax) * invSum;
  }
}

// ---------- Naive three-pass softmax, f32 ----------
// Pass 1: max via warp/block reduce
// Pass 2: write exp(xi - max) to output, accumulate sum
// Pass 3: divide output by sum
template <int BLOCK_SIZE>
__global__ void softmaxNaiveF32Kernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int64_t cols) {
  constexpr int kNumWarps = BLOCK_SIZE / kWarpSize;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warpId = tid / kWarpSize;
  const int laneId = tid & (kWarpSize - 1);

  const float* rowIn = x + row * cols;
  float* rowOut = y + row * cols;

  // Pass 1: max.
  float localMax = kNegInfIdentity;
  for (int64_t i = tid; i < cols; i += BLOCK_SIZE) {
    localMax = fmaxf(localMax, rowIn[i]);
  }
  float m = warpReduceMax(localMax);
  __shared__ float sMax[kWarpSize];
  if (laneId == 0) sMax[warpId] = m;
  __syncthreads();
  if (warpId == 0) {
    m = (laneId < kNumWarps) ? sMax[laneId] : kNegInfIdentity;
    m = warpReduceMax(m);
    if (laneId == 0) sMax[0] = m;
  }
  __syncthreads();
  const float rowMax = sMax[0];

  // Pass 2: exp and sum. Write exps to output, keep sum in register.
  float localSum = 0.f;
  for (int64_t i = tid; i < cols; i += BLOCK_SIZE) {
    float e = __expf(rowIn[i] - rowMax);
    rowOut[i] = e;
    localSum += e;
  }
  float s = warpReduceSum(localSum);
  __shared__ float sSum[kWarpSize];
  if (laneId == 0) sSum[warpId] = s;
  __syncthreads();
  if (warpId == 0) {
    s = (laneId < kNumWarps) ? sSum[laneId] : 0.f;
    s = warpReduceSum(s);
    if (laneId == 0) sSum[0] = s;
  }
  __syncthreads();
  const float invSum = 1.f / sSum[0];

  // Pass 3: normalize.
  for (int64_t i = tid; i < cols; i += BLOCK_SIZE) {
    rowOut[i] = rowOut[i] * invSum;
  }
}

// ---------- Online softmax, f16 ----------
// f16 storage, f32 accumulation. exp() is always f32 for numerical stability.
template <int BLOCK_SIZE>
__global__ void softmaxOnlineF16Kernel(const __half* __restrict__ x,
                                       __half* __restrict__ y,
                                       int64_t cols) {
  constexpr int kNumWarps = BLOCK_SIZE / kWarpSize;
  const int64_t row = blockIdx.x;
  const int tid = threadIdx.x;
  const int warpId = tid / kWarpSize;
  const int laneId = tid & (kWarpSize - 1);

  const __half* rowIn = x + row * cols;
  __half* rowOut = y + row * cols;

  float m = kNegInfIdentity;
  float d = 0.f;
  for (int64_t i = tid; i < cols; i += BLOCK_SIZE) {
    float xi = __half2float(rowIn[i]);
    float mNew = fmaxf(m, xi);
    d = d * __expf(m - mNew) + __expf(xi - mNew);
    m = mNew;
  }
  warpMergeOnline(m, d);

  __shared__ float sMax[kWarpSize];
  __shared__ float sSum[kWarpSize];
  if (laneId == 0) { sMax[warpId] = m; sSum[warpId] = d; }
  __syncthreads();
  if (warpId == 0) {
    m = (laneId < kNumWarps) ? sMax[laneId] : kNegInfIdentity;
    d = (laneId < kNumWarps) ? sSum[laneId] : 0.f;
    warpMergeOnline(m, d);
    if (laneId == 0) { sMax[0] = m; sSum[0] = d; }
  }
  __syncthreads();

  const float rowMax = sMax[0];
  const float invSum = 1.f / sSum[0];
  for (int64_t i = tid; i < cols; i += BLOCK_SIZE) {
    float xi = __half2float(rowIn[i]);
    rowOut[i] = __float2half(__expf(xi - rowMax) * invSum);
  }
}

// Pick a block size class based on column count. Power-of-two widths are
// preferred so the block-size specialization stays cache-efficient.
template <typename Launcher>
int dispatchBlockSize(int64_t cols, Launcher& launcher) {
  if (cols <= 64)   return launcher.template launch<32>();
  if (cols <= 128)  return launcher.template launch<64>();
  if (cols <= 256)  return launcher.template launch<128>();
  if (cols <= 1024) return launcher.template launch<256>();
  return launcher.template launch<512>();
}

struct OnlineF32Launcher {
  const float* x;
  float* y;
  int64_t rows, cols;
  cudaStream_t stream;
  template <int BS>
  int launch() {
    softmaxOnlineF32Kernel<BS>
        <<<static_cast<unsigned int>(rows), BS, 0, stream>>>(x, y, cols);
    return static_cast<int>(cudaPeekAtLastError());
  }
};

struct NaiveF32Launcher {
  const float* x;
  float* y;
  int64_t rows, cols;
  cudaStream_t stream;
  template <int BS>
  int launch() {
    softmaxNaiveF32Kernel<BS>
        <<<static_cast<unsigned int>(rows), BS, 0, stream>>>(x, y, cols);
    return static_cast<int>(cudaPeekAtLastError());
  }
};

struct OnlineF16Launcher {
  const __half* x;
  __half* y;
  int64_t rows, cols;
  cudaStream_t stream;
  template <int BS>
  int launch() {
    softmaxOnlineF16Kernel<BS>
        <<<static_cast<unsigned int>(rows), BS, 0, stream>>>(x, y, cols);
    return static_cast<int>(cudaPeekAtLastError());
  }
};

}  // namespace

extern "C" int mlc_softmax_online_f32(const float* d_input, float* d_output,
                                      int64_t rows, int64_t cols,
                                      void* cuda_stream) {
  if (rows <= 0 || cols <= 0) return 0;
  OnlineF32Launcher l{d_input, d_output, rows, cols,
                       static_cast<cudaStream_t>(cuda_stream)};
  return dispatchBlockSize(cols, l);
}

extern "C" int mlc_softmax_naive_f32(const float* d_input, float* d_output,
                                     int64_t rows, int64_t cols,
                                     void* cuda_stream) {
  if (rows <= 0 || cols <= 0) return 0;
  NaiveF32Launcher l{d_input, d_output, rows, cols,
                      static_cast<cudaStream_t>(cuda_stream)};
  return dispatchBlockSize(cols, l);
}

extern "C" int mlc_softmax_online_f16(const void* d_input, void* d_output,
                                      int64_t rows, int64_t cols,
                                      void* cuda_stream) {
  if (rows <= 0 || cols <= 0) return 0;
  OnlineF16Launcher l{static_cast<const __half*>(d_input),
                       static_cast<__half*>(d_output),
                       rows, cols,
                       static_cast<cudaStream_t>(cuda_stream)};
  return dispatchBlockSize(cols, l);
}

extern "C" const char* mlc_softmax_online_last_error(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
