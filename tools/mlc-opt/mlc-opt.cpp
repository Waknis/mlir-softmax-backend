#include "mlc/Passes.h"

#if __has_include("mlir/Dialect/Arith/IR/Arith.h")
#include "mlir/Dialect/Arith/IR/Arith.h"
#else
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();

  mlc::registerMLCPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLC optimizer driver\n", registry));
}
