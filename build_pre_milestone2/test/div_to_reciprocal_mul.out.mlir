module {
  func.func @softmax_tail(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: f32) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 1.000000e+00 : f32
    %0 = arith.divf %cst, %arg2 : f32
    scf.for %arg3 = %c0 to %c128 step %c1 {
      %1 = memref.load %arg0[%arg3] : memref<128xf32>
      %2 = arith.mulf %1, %0 : f32
      memref.store %2, %arg1[%arg3] : memref<128xf32>
    }
    return
  }
}

