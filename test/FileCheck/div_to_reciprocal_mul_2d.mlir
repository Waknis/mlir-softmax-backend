// RUN: mlc-opt --mlc-div-to-reciprocal-mul %s | FileCheck %s

// Verifies the div->reciprocal*mul pass correctly hoists the reciprocal out
// of a nested (2D) loop. The outer-loop-variant `%sum` is re-loaded per row
// but is inner-loop-invariant, so the reciprocal should be computed once
// per row and the inner-loop `divf` replaced by `mulf`.

module {
  func.func @rowwise_divide(%input: memref<16x32xf32>,
                             %sums:  memref<16xf32>,
                             %output: memref<16x32xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cM = arith.constant 16 : index
    %cN = arith.constant 32 : index
    scf.for %i = %c0 to %cM step %c1 {
      %sum = memref.load %sums[%i] : memref<16xf32>
      scf.for %j = %c0 to %cN step %c1 {
        %x = memref.load %input[%i, %j] : memref<16x32xf32>
        %y = arith.divf %x, %sum : f32
        memref.store %y, %output[%i, %j] : memref<16x32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @rowwise_divide
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: %[[SUM:.*]] = memref.load %arg1[%[[I]]] : memref<16xf32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[RECIP:.*]] = arith.divf %[[ONE]], %[[SUM]] : f32
// CHECK: scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT: arith.divf %{{.*}}, %[[SUM]]
// CHECK: %[[X:.*]] = memref.load %arg0[%[[I]], %[[J]]] : memref<16x32xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[X]], %[[RECIP]] : f32
// CHECK: memref.store %[[MUL]], %arg2[%[[I]], %[[J]]] : memref<16x32xf32>
