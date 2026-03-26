module {
  func.func @softmax_norm(%input: memref<1024xf32>,
                          %output: memref<1024xf32>,
                          %sum: f32) {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c1024 step %c1 {
      %x = memref.load %input[%i] : memref<1024xf32>
      %y = arith.divf %x, %sum : f32
      memref.store %y, %output[%i] : memref<1024xf32>
    }
    return
  }
}
