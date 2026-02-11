module {
  func.func @softmax_norm(%sum: f32) -> f32 {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %one = arith.constant 1.0 : f32
    %acc = scf.for %i = %c0 to %c1024 step %c1 iter_args(%s = %zero) -> (f32) {
      %d = arith.divf %one, %sum : f32
      %next = arith.addf %s, %d : f32
      scf.yield %next : f32
    }
    return %acc : f32
  }
}
