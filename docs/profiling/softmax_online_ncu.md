# Nsight Compute profile: hand-written online softmax

Shape: 4096 x 4096, f32
Hardware: NVIDIA GeForce RTX 5060 Ti (SM 12.0, GDDR7)
Tool: `ncu 2026.1.1.0`, sections `SpeedOfLight`, `MemoryWorkloadAnalysis`,
`LaunchStats`, `Occupancy`, `WarpStateStats`.

This profile is for the hand CUDA / Triton baseline lane, not for an
MLIR-emitted kernel. It is kept as performance evidence for future MLIR
softmax work.

Raw report: [`softmax_online_f32_4k.ncu-rep`](softmax_online_f32_4k.ncu-rep)

## Speed of Light

| Metric                  | Value           |
|-------------------------|-----------------|
| DRAM Throughput         | **87.36%**      |
| Memory Throughput       | 87.36%          |
| Compute (SM) Throughput | 17.56%          |
| Duration                | 328.8 us        |
| Grid size               | 4096 blocks     |
| Block size              | 512 threads     |
| Achieved Occupancy      | ~92%            |

> Memory throughput is >80% of the device's SoL; ncu flags this as a
> DRAM-bound workload. Compute throughput at 17.6% confirms there is no
> realistic avenue for further speedup by reducing arithmetic —
> softmax is fundamentally bandwidth-bound at this shape.

## Stall reasons

- **45.8 of 60.2 avg cycles-between-issues** spent stalled on L1TEX
  scoreboard (i.e. waiting for a global load to return). This is the
  expected memory-bound stall profile — coalesced loads + reductions means
  every warp spends most of its time waiting on DRAM.
- **L2 Hit Rate**: 28.16%. The second traversal (normalize pass,
  `y = exp(xi - m) / d`) re-reads the input row `xi`, hitting L2 for rows
  small enough (≤1 MB) and missing for the rest.
- **L1/TEX Hit Rate**: 7.09%. Expected; each thread accesses a strided
  element per pass with low temporal reuse within a warp.

## Comparison across backends at the same shape

| Backend            | DRAM %peak | Duration   | SM %peak | Achieved Occ. |
|--------------------|-----------:|-----------:|---------:|--------------:|
| `hand_online_f32`  |     87.36% |   328.8 us |   17.56% |        ~92%   |
| `hand_naive_f32`   |     84.29% |   354.9 us |   22.76% |        92.56% |
| `hand_online_f16`  |     88.11% |   145.1 us |   41.61% |        92.21% |
| `triton_f32`       |     88.13% |   315.3 us |   14.12% |        91.07% |

Source reports in this directory:
[`softmax_naive_f32_4k.ncu-rep`](softmax_naive_f32_4k.ncu-rep),
[`softmax_online_f16_4k.ncu-rep`](softmax_online_f16_4k.ncu-rep),
[`softmax_triton_f32_4k.ncu-rep`](softmax_triton_f32_4k.ncu-rep).

All four kernels cluster within 4 percentage points of each other at 84-88%
of peak DRAM bandwidth, confirming that on this shape:

1. The bandwidth ceiling dominates algorithmic choice. Online vs three-pass
   naive, Triton vs hand-CUDA — the absolute difference on 4096x4096 is
   <10% of per-launch wall time.
2. The online (fused) variant's structural advantage (2 row traversals vs
   3) is partly neutralized by L2 hits on the second pass for typical
   shapes; see ncu's 28% L2 hit rate. At shapes where `y` exceeds L2, the
   fused kernel's bandwidth advantage widens.
3. f16 storage halves the bytes moved and lands at 145 us — close to the
   theoretical 2x speedup over the f32 variant, limited only by the
   marginally lower absolute DRAM throughput on narrower data widths.

## How to reproduce

```bash
python -m pip install -e ".[dev,triton]"
cmake --build build --target mlc_softmax_kernels

MLC_SOFTMAX_KERNELS_LIB=kernels/libmlc_softmax_kernels.so \
  ncu --launch-skip 3 --launch-count 1 \
      --section SpeedOfLight \
      --section MemoryWorkloadAnalysis \
      --section LaunchStats \
      --section Occupancy \
      --section WarpStateStats \
      --export docs/profiling/softmax_online_f32_4k --force-overwrite \
      python -m benchmarks.profile_one --backend online_f32 --shape 4096x4096
```
