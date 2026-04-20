// RUN: mlc-opt --mlc-div-to-reciprocal-mul %s | FileCheck %s

module {
  func.func @softmax_tail(%input: memref<128xf32>, %output: memref<128xf32>,
                          %sum: f32) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c128 step %c1 {
      %x = memref.load %input[%i] : memref<128xf32>
      %y = arith.divf %x, %sum : f32
      memref.store %y, %output[%i] : memref<128xf32>
    }
    return
  }

  func.func @keeps_loop_variant_denominator(%input: memref<128xf32>,
                                            %output: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c128 step %c1 {
      %x = memref.load %input[%i] : memref<128xf32>
      %denom = arith.addf %x, %x : f32
      %y = arith.divf %x, %denom : f32
      memref.store %y, %output[%i] : memref<128xf32>
    }
    return
  }
}

// CHECK-LABEL: func.func @softmax_tail
// CHECK: %[[ONE:.*]] = arith.constant {{.*}} : f32
// CHECK: %[[RECIP:.*]] = arith.divf %[[ONE]], %arg2 : f32
// CHECK: scf.for
// CHECK-NOT: arith.divf %{{.*}}, %arg2 : f32
// CHECK: %[[LOAD:.*]] = memref.load %arg0[%{{.*}}] : memref<128xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[LOAD]], %[[RECIP]] : f32
// CHECK: memref.store %[[MUL]], %arg1[%{{.*}}] : memref<128xf32>

// CHECK-LABEL: func.func @keeps_loop_variant_denominator
// CHECK: scf.for
// CHECK: %[[X:.*]] = memref.load %arg0[%{{.*}}] : memref<128xf32>
// CHECK: %[[DENOM:.*]] = arith.addf %[[X]], %[[X]] : f32
// CHECK: %[[DIV:.*]] = arith.divf %[[X]], %[[DENOM]] : f32
// CHECK: memref.store %[[DIV]], %arg1[%{{.*}}] : memref<128xf32>
