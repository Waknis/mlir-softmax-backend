# Softmax benchmark results

- Device: NVIDIA GeForce RTX 5060 Ti (SM 12.0)
- torch: 2.11.0+cu128 / CUDA 12.8
- Peak memory bandwidth assumed: 448 GB/s

Row-wise softmax along the inner axis. Time is median over 200 CUDA-event iterations after 50 warmup iters.

These are baseline measurements for the hand CUDA and Triton kernels. They
are not emitted by the current MLIR pipeline; they are the performance
target for future MLIR-generated row-wise softmax.

| Shape | Dtype | Backend | ms (med) | ms (p95) | GB/s | % peak |
|-------|-------|---------|---------:|---------:|-----:|-------:|
| 256x256 | f32 | `hand_online_f32` | 0.0062 | 0.0068 | 84.7 | 18.9% |
| 256x256 | f32 | `hand_naive_f32` | 0.0067 | 0.0072 | 78.8 | 17.6% |
| 256x256 | f16 | `hand_online_f16` | 0.0051 | 0.0068 | 51.7 | 11.5% |
| 256x256 | f32 | `triton_f32` | 0.0047 | 0.0066 | 110.7 | 24.7% |
| 512x1024 | f32 | `hand_online_f32` | 0.0108 | 0.0128 | 388.9 | 86.8% |
| 512x1024 | f32 | `hand_naive_f32` | 0.0128 | 0.0166 | 326.9 | 73.0% |
| 512x1024 | f16 | `hand_online_f16` | 0.0088 | 0.0108 | 237.4 | 53.0% |
| 512x1024 | f32 | `triton_f32` | 0.0107 | 0.0128 | 391.8 | 87.5% |
| 1024x1024 | f32 | `hand_online_f32` | 0.0230 | 0.0287 | 364.1 | 81.3% |
| 1024x1024 | f32 | `hand_naive_f32` | 0.0275 | 0.0334 | 304.6 | 68.0% |
| 1024x1024 | f16 | `hand_online_f16` | 0.0150 | 0.0175 | 280.1 | 62.5% |
| 1024x1024 | f32 | `triton_f32` | 0.0213 | 0.0287 | 394.2 | 88.0% |
| 1024x4096 | f32 | `hand_online_f32` | 0.0906 | 0.0942 | 370.5 | 82.7% |
| 1024x4096 | f32 | `hand_naive_f32` | 0.0932 | 0.0983 | 360.0 | 80.3% |
| 1024x4096 | f16 | `hand_online_f16` | 0.0471 | 0.0512 | 355.9 | 79.4% |
| 1024x4096 | f32 | `triton_f32` | 0.0885 | 0.0927 | 379.1 | 84.6% |
| 2048x2048 | f32 | `hand_online_f32` | 0.0907 | 0.0962 | 369.9 | 82.6% |
| 2048x2048 | f32 | `hand_naive_f32` | 0.0926 | 0.1029 | 362.5 | 80.9% |
| 2048x2048 | f16 | `hand_online_f16` | 0.0497 | 0.0532 | 337.6 | 75.4% |
| 2048x2048 | f32 | `triton_f32` | 0.0867 | 0.0934 | 386.9 | 86.4% |
| 4096x4096 | f32 | `hand_online_f32` | 0.3529 | 0.5964 | 380.3 | 84.9% |
| 4096x4096 | f32 | `hand_naive_f32` | 0.3589 | 0.6079 | 373.9 | 83.5% |
| 4096x4096 | f16 | `hand_online_f16` | 0.1807 | 0.2012 | 371.3 | 82.9% |
| 4096x4096 | f32 | `triton_f32` | 0.3487 | 0.3979 | 384.9 | 85.9% |
| 8192x8192 | f32 | `hand_online_f32` | 1.4219 | 1.7546 | 377.6 | 84.3% |
| 8192x8192 | f32 | `hand_naive_f32` | 1.4280 | 1.8048 | 376.0 | 83.9% |
| 8192x8192 | f16 | `hand_online_f16` | 0.7090 | 0.9988 | 378.6 | 84.5% |
| 8192x8192 | f32 | `triton_f32` | 1.3906 | 1.6831 | 386.1 | 86.2% |
