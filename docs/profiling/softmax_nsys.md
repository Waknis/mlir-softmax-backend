# Nsight Systems timeline: softmax benchmark sweep

Traced run: `python -m benchmarks.softmax_gpu_bench --shapes 4096x4096 --iters 50`

This trace covers the non-MLIR baseline softmax kernels. The current MLIR
pipeline does not emit these kernels yet.

Raw report: [`softmax_sweep_nsys.nsys-rep`](softmax_sweep_nsys.nsys-rep)
(open in Nsight Systems UI for the full timeline view; regenerate the
`.sqlite` companion for scripted queries with
`nsys export --type sqlite softmax_sweep_nsys.nsys-rep`).

## Kernel summary (sum over the run, 4096x4096, 50 timed iters + 50 warmup per
backend)

| Kernel                                          | Calls | Avg (ns) | Median (ns) | % GPU time |
|-------------------------------------------------|------:|---------:|------------:|-----------:|
| `_softmax_triton_kernel`                        |   101 |  372,392 |     346,443 |     22.2%  |
| `softmaxOnlineF32Kernel<512>`                   |   101 |  368,196 |     349,358 |     21.9%  |
| `softmaxNaiveF32Kernel<512>`                    |   101 |  366,930 |     355,479 |     21.9%  |
| `softmaxOnlineF16Kernel<512>`                   |   101 |  188,716 |     175,304 |     11.2%  |
| `FillFunctor<signed char>` (L2 flush)           |   200 |  160,866 |     148,229 |     19.0%  |

The "FillFunctor" line is the harness's inter-iter L2-flush write and is
what keeps the per-launch bandwidth numbers honest — skip `--no-flush-l2`
and that 19% of GPU time disappears, at the cost of hot-L2 measurements on
small shapes.

## Cross-check with ncu

The nsys-reported `softmaxOnlineF32Kernel` median of 349 us matches the
ncu single-launch capture of 328 us within 5%; the delta is the range
across launches that nsys averages over (the p95 in the harness table is
also ~375 us, consistent).

## How to reproduce

```bash
MLC_SOFTMAX_KERNELS_LIB=kernels/libmlc_softmax_kernels.so \
  nsys profile --trace=cuda --stats=true --force-overwrite true \
      --output docs/profiling/softmax_sweep_nsys \
      python -m benchmarks.softmax_gpu_bench --shapes 4096x4096 --iters 50
```
