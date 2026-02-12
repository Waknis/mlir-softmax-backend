#include "mlc/Passes.h"

#if __has_include("mlir/Dialect/Arith/IR/Arith.h")
#include "mlir/Dialect/Arith/IR/Arith.h"
#else
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace mlc {
namespace {

class DivToReciprocalMulPass
    : public mlir::PassWrapper<DivToReciprocalMulPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DivToReciprocalMulPass)

  mlir::StringRef getArgument() const final {
    return "mlc-div-to-reciprocal-mul";
  }

  mlir::StringRef getDescription() const final {
    return "Hoist loop-invariant divisors and replace divf with mulf by "
           "reciprocal.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithmeticDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final {
    auto module = getOperation();
    module.walk([&](mlir::scf::ForOp forOp) { rewriteLoop(forOp); });
  }

 private:
  void rewriteLoop(mlir::scf::ForOp forOp) {
    llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::arith::DivFOp>> divsByRhs;

    forOp.getBody()->walk([&](mlir::arith::DivFOp divOp) {
      mlir::Value denom = divOp.getRhs();
      if (!denom.getType().isa<mlir::FloatType>()) {
        return;
      }
      if (!forOp.isDefinedOutsideOfLoop(denom)) {
        return;
      }
      divsByRhs[denom].push_back(divOp);
    });

    if (divsByRhs.empty()) {
      return;
    }

    mlir::OpBuilder outerBuilder(forOp);
    for (auto &entry : divsByRhs) {
      mlir::Value denom = entry.first;
      auto *divOps = &entry.second;
      if (divOps->empty()) {
        continue;
      }

      auto floatType = denom.getType().dyn_cast<mlir::FloatType>();
      if (!floatType) {
        continue;
      }

      mlir::Value one = outerBuilder.create<mlir::arith::ConstantOp>(
          forOp.getLoc(), mlir::FloatAttr::get(floatType, 1.0));
      mlir::Value reciprocal =
          outerBuilder.create<mlir::arith::DivFOp>(forOp.getLoc(), one, denom);

      for (mlir::arith::DivFOp divOp : *divOps) {
        mlir::OpBuilder builder(divOp);
        mlir::Value mul = builder.create<mlir::arith::MulFOp>(
            divOp.getLoc(), divOp.getLhs(), reciprocal);
        divOp.replaceAllUsesWith(mul);
        divOp.erase();
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createDivToReciprocalMulPass() {
  return std::make_unique<DivToReciprocalMulPass>();
}

void registerMLCPasses() { mlir::PassRegistration<DivToReciprocalMulPass>(); }

}  // namespace mlc
