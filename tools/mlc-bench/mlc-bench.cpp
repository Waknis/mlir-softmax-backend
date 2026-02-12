#include "mlc/Passes.h"

#if __has_include("mlir/Dialect/Arith/IR/Arith.h")
#include "mlir/Dialect/Arith/IR/Arith.h"
#else
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#if __has_include("mlir/Parser/Parser.h")
#include "mlir/Parser/Parser.h"
#else
#include "mlir/Parser.h"
#endif
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace {

llvm::cl::opt<std::string> kSizes(
    "sizes", llvm::cl::desc("Comma-separated loop sizes."), llvm::cl::init("1024"));
llvm::cl::opt<unsigned> kIters(
    "iters", llvm::cl::desc("Benchmark iterations per size."), llvm::cl::init(20));

std::string buildLoopModule(unsigned size) {
  std::ostringstream oss;
  oss << "module {\n"
      << "  func.func @softmax_norm(%sum: f32) -> f32 {\n"
      << "    %c0 = arith.constant 0 : index\n"
      << "    %c" << size << " = arith.constant " << size << " : index\n"
      << "    %c1 = arith.constant 1 : index\n"
      << "    %zero = arith.constant 0.0 : f32\n"
      << "    %one = arith.constant 1.0 : f32\n"
      << "    %acc = scf.for %i = %c0 to %c" << size
      << " step %c1 iter_args(%s = %zero) -> (f32) {\n"
      << "      %d = arith.divf %one, %sum : f32\n"
      << "      %next = arith.addf %s, %d : f32\n"
      << "      scf.yield %next : f32\n"
      << "    }\n"
      << "    return %acc : f32\n"
      << "  }\n"
      << "}\n";
  return oss.str();
}

struct DivStats {
  std::uint64_t outsideLoop = 0;
  std::uint64_t inLoopBody = 0;
};

DivStats countDivs(mlir::ModuleOp module) {
  DivStats stats;
  module.walk([&](mlir::arith::DivFOp divOp) {
    if (divOp->getParentOfType<mlir::scf::ForOp>()) {
      ++stats.inLoopBody;
    } else {
      ++stats.outsideLoop;
    }
  });
  return stats;
}

mlir::OwningOpRef<mlir::ModuleOp> parseModuleText(mlir::MLIRContext &context,
                                                  llvm::StringRef source) {
  return mlir::parseSourceString<mlir::ModuleOp>(source, &context);
}

std::vector<unsigned> parseSizes() {
  llvm::SmallVector<llvm::StringRef> rawSizes;
  llvm::SplitString(llvm::StringRef(kSizes), rawSizes, ",");
  std::vector<unsigned> sizes;
  sizes.reserve(rawSizes.size());
  for (llvm::StringRef token : rawSizes) {
    token = token.trim();
    if (token.empty()) {
      continue;
    }
    unsigned value = 0;
    if (token.getAsInteger(10, value)) {
      llvm::errs() << "Invalid size: " << token << "\n";
      continue;
    }
    if (value == 0) {
      llvm::errs() << "Skipping zero size entry.\n";
      continue;
    }
    sizes.push_back(value);
  }
  if (sizes.empty()) {
    sizes.push_back(1024);
  }
  return sizes;
}

}  // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "MLC pass benchmark\n");

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);

  std::vector<unsigned> sizes = parseSizes();

  llvm::outs() << "size\tbaseline_est_divs\toptimized_est_divs\tdiv_reduction_%\t"
                  "avg_pass_ms\n";
  for (unsigned size : sizes) {
    std::string moduleText = buildLoopModule(size);
    auto baselineModule = parseModuleText(context, moduleText);
    if (!baselineModule) {
      llvm::errs() << "Failed to parse baseline module for size " << size << "\n";
      return 1;
    }

    DivStats baselineStats = countDivs(*baselineModule);
    std::uint64_t baselineEstDivs =
        baselineStats.outsideLoop + baselineStats.inLoopBody * size;

    std::chrono::duration<double, std::milli> totalMs(0.0);
    mlir::OwningOpRef<mlir::ModuleOp> finalModule;
    for (unsigned iter = 0; iter < kIters; ++iter) {
      auto module = parseModuleText(context, moduleText);
      if (!module) {
        llvm::errs() << "Failed to parse benchmark module at iteration " << iter
                     << "\n";
        return 1;
      }

      mlir::PassManager pm(&context);
      pm.addPass(mlc::createDivToReciprocalMulPass());

      auto start = std::chrono::high_resolution_clock::now();
      mlir::LogicalResult result = pm.run(*module);
      auto end = std::chrono::high_resolution_clock::now();
      if (mlir::failed(result)) {
        llvm::errs() << "Pass pipeline failed for size " << size << "\n";
        return 1;
      }
      totalMs += end - start;
      if (iter + 1 == kIters) {
        finalModule = std::move(module);
      }
    }

    DivStats optimizedStats = countDivs(*finalModule);
    std::uint64_t optimizedEstDivs =
        optimizedStats.outsideLoop + optimizedStats.inLoopBody * size;
    double reduction = baselineEstDivs == 0
                           ? 0.0
                           : 100.0 *
                                 (1.0 - (static_cast<double>(optimizedEstDivs) /
                                         static_cast<double>(baselineEstDivs)));
    double avgMs = totalMs.count() / static_cast<double>(kIters);

    llvm::outs() << size << "\t" << baselineEstDivs << "\t\t"
                 << optimizedEstDivs << "\t\t" << reduction << "\t\t" << avgMs
                 << "\n";
  }

  return 0;
}
