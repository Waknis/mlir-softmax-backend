"""Generate the roofline plot in docs/profiling/roofline.png.

Requires matplotlib (optional dev dep). Numbers come from:
- Peak compute: RTX 5060 Ti single-precision FMA throughput (datasheet, ~24
  TFLOP/s at boost). f16 in CUDA cores is 2x that, ~48 TFLOP/s.
- Peak DRAM BW: 448 GB/s (GDDR7 128-bit).
- Measured operating points: from `docs/results.json` (4096x4096 row) and
  ncu measurements at the same shape.

Softmax per-element arithmetic intensity (AI) = FLOPs / bytes moved:
  Online fused: ~5 FLOPs/elem (max, exp, mul, div, add merged), 2 reads +
  1 write per element -> 12 bytes/elem for f32 -> AI = 5/12 = 0.42 FLOP/byte
  (lower for naive 3-pass because of the extra in-flight traffic).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PEAK_COMPUTE_F32 = 24.0e12   # FLOP/s
PEAK_COMPUTE_F16 = 48.0e12   # FLOP/s (FP16 in CUDA cores, no tensor cores)
PEAK_BW          = 448.0e9   # B/s

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = json.loads((ROOT / "docs" / "results.json").read_text())


def _flops_per_elem(backend: str) -> float:
    # Online softmax (2 traversals): per elem do 1 cmp+expf+fma in pass 1,
    # 1 sub+expf+mul in pass 2 ~ 7 ops. Naive (3 traversals): cmp in p1,
    # sub+expf+add in p2, mul in p3 ~ 6 ops. Triton online-ish: ~6.
    if backend.startswith("hand_online"):
        return 7.0
    return 6.0


def _bytes_per_elem(backend: str) -> float:
    # Elemsize (f16 backends move 2 bytes per read/write).
    elem = 2 if "f16" in backend else 4
    # Online: 1 read + 1 write per elem over 2 passes -> 4 read + 2 write
    # logically; L2/coalesce brings it closer to 1 read + 1 write DRAM in
    # practice for rows smaller than L2. Naive 3-pass traverses 3x. We use
    # the theoretical DRAM-optimal traffic (2 R + 1 W for online = 12 bytes
    # f32; 3 R + 2 W naive = 20 bytes f32) as the AI denominator, which is
    # what the benchmark reports against anyway (2 * rows * cols * elemsize).
    return 2.0 * elem


def arithmetic_intensity(backend: str) -> float:
    return _flops_per_elem(backend) / _bytes_per_elem(backend)


def achieved_flops(entry: dict) -> float:
    # entry["bandwidth_gbs"] * 1e9 = bytes/sec (at the DRAM measurement).
    # FLOPs/sec = bytes/sec * (flops/elem) / (bytes/elem).
    bw = entry["bandwidth_gbs"] * 1e9
    return bw * _flops_per_elem(entry["backend"]) / _bytes_per_elem(entry["backend"])


def main() -> None:
    # Pick the 4096x4096 rows -- the shape that fits > L2 and gives the
    # cleanest bandwidth-bound reading.
    picks = [r for r in RESULTS if r["shape"] == "4096x4096"]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    ai = np.logspace(-2, 2, 200)
    peak_mem_f32 = PEAK_BW * ai            # FLOP/s ceiling from bandwidth * AI
    peak_compute_f32 = np.full_like(ai, PEAK_COMPUTE_F32)
    peak_compute_f16 = np.full_like(ai, PEAK_COMPUTE_F16)

    ax.plot(ai, np.minimum(peak_mem_f32, peak_compute_f32) / 1e12,
            color="tab:blue", label="Roofline (f32, CUDA cores)")
    ax.plot(ai, np.minimum(peak_mem_f32, peak_compute_f16) / 1e12,
            color="tab:orange", linestyle="--", label="Roofline (f16, CUDA cores)")

    markers = {
        "hand_online_f32": ("o", "tab:blue"),
        "hand_naive_f32":  ("s", "tab:cyan"),
        "hand_online_f16": ("D", "tab:orange"),
        "triton_f32":      ("^", "tab:green"),
    }
    for r in picks:
        m, c = markers.get(r["backend"], ("x", "black"))
        x = arithmetic_intensity(r["backend"])
        y = achieved_flops(r) / 1e12
        ax.scatter([x], [y], marker=m, color=c, s=90, edgecolor="black",
                   zorder=5, label=r["backend"])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOPs / byte)")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title("Softmax roofline — RTX 5060 Ti, 4096x4096")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(1e-2, 1e2)

    out = ROOT / "docs" / "profiling" / "roofline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
