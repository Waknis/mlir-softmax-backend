// Row-wise softmax *normalization* step (per-row divide-by-sum).
//
// Scope note: this is the DIVIDE stage of softmax (y[i,j] = x[i,j] / sum[i]),
// not the full algorithm. We keep the MLIR example at the divide stage
// because:
//
//   1. It is what the existing lowering pipeline (arith+func+memref+scf ->
//      LLVM -> PTX) is dialect-complete for. Adding `math.exp` would
//      require wiring in the math-to-LLVM conversion, which is future work.
//   2. This is also the op that the custom `mlc-div-to-reciprocal-mul`
//      pass optimizes, so this file doubles as the pass's input.
//
// Shape: 1024 rows x 1024 cols, nested scf.for (2-D variant of the 1-D
// `softmax.mlir`). The per-row `sum` is loop-invariant inside the inner
// loop, so the div-to-reciprocal-mul pass hoists the reciprocal to just
// inside the outer loop.

module {
  func.func @softmax_rowwise_normalize(%input: memref<1024x1024xf32>,
                                        %sums:  memref<1024xf32>,
                                        %output: memref<1024x1024xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cM = arith.constant 1024 : index
    %cN = arith.constant 1024 : index
    scf.for %i = %c0 to %cM step %c1 {
      %sum = memref.load %sums[%i] : memref<1024xf32>
      scf.for %j = %c0 to %cN step %c1 {
        %x = memref.load %input[%i, %j] : memref<1024x1024xf32>
        %y = arith.divf %x, %sum : f32
        memref.store %y, %output[%i, %j] : memref<1024x1024xf32>
      }
    }
    return
  }
}
