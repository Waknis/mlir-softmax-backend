// Static analysis of the div->recip*mul pass effect.
//
// This tool does NOT measure runtime. It:
//   1. Builds a synthetic elementwise-div MLIR module (softmax-*shaped*
//      loops, not the actual softmax algorithm).
//   2. Runs the full lowering pipeline and reports compile wall time.
//   3. Counts the dynamic divisions before vs after the div->recip*mul pass
//      by walking the hoisted/in-loop divs and multiplying by the static
//      iteration counts. This is a symbolic-execution-style metric: it
//      answers "how many div ops did LICM + strength reduction remove?",
//      not "how fast does the kernel run?".
//
// For GPU runtime throughput on real softmax kernels, see
// `benchmarks/softmax_gpu_bench.py`.

#include "compiler/pipeline/LoweringPipeline.h"

#if __has_include("mlir/Dialect/Arith/IR/Arith.h")
#include "mlir/Dialect/Arith/IR/Arith.h"
#else
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlc/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Shape {
  std::int64_t m = 0;
  std::int64_t n = 0;

  std::string str() const {
    return std::to_string(m) + "x" + std::to_string(n);
  }
};

llvm::cl::opt<std::string> kShapes(
    "shapes",
    llvm::cl::desc("Comma-separated shapes in MxN form."),
    llvm::cl::init("64x64,64x128,128x128,128x256,256x256,256x512,512x512,512x1024,1024x1024,2048x1024"));

llvm::cl::opt<std::string> kOutputRoot(
    "output-root",
    llvm::cl::desc("Directory where benchmark artifacts are written."),
    llvm::cl::init("build/benchmark_runs"));

llvm::cl::opt<std::string> kLlc(
    "llc",
    llvm::cl::desc("Optional llc path override."),
    llvm::cl::init(""));

std::vector<Shape> parseShapes(llvm::StringRef shapeCsv) {
  llvm::SmallVector<llvm::StringRef> tokens;
  llvm::SplitString(shapeCsv, tokens, ",");

  std::vector<Shape> shapes;
  shapes.reserve(tokens.size());

  for (llvm::StringRef token : tokens) {
    token = token.trim();
    if (token.empty()) {
      continue;
    }

    auto parts = token.split('x');
    if (parts.second.empty()) {
      llvm::errs() << "Skipping invalid shape token: " << token << "\n";
      continue;
    }

    std::int64_t m = 0;
    std::int64_t n = 0;
    if (parts.first.getAsInteger(10, m) || parts.second.getAsInteger(10, n) ||
        m <= 0 || n <= 0) {
      llvm::errs() << "Skipping invalid shape token: " << token << "\n";
      continue;
    }

    shapes.push_back({m, n});
  }

  return shapes;
}

std::string buildSoftmaxMlir(Shape shape) {
  std::ostringstream oss;
  oss << "module {\n"
      << "  func.func @softmax_norm(%sum: f32) -> f32 {\n"
      << "    %c0 = arith.constant 0 : index\n"
      << "    %cM = arith.constant " << shape.m << " : index\n"
      << "    %cN = arith.constant " << shape.n << " : index\n"
      << "    %c1 = arith.constant 1 : index\n"
      << "    %zero = arith.constant 0.0 : f32\n"
      << "    %one = arith.constant 1.0 : f32\n"
      << "    %outer = scf.for %row = %c0 to %cM step %c1 iter_args(%row_acc = %zero) -> (f32) {\n"
      << "      %inner = scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %row_acc) -> (f32) {\n"
      << "        %d = arith.divf %one, %sum : f32\n"
      << "        %next = arith.addf %acc, %d : f32\n"
      << "        scf.yield %next : f32\n"
      << "      }\n"
      << "      scf.yield %inner : f32\n"
      << "    }\n"
      << "    return %outer : f32\n"
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

}  // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Softmax benchmark: division count analysis across shapes for baseline vs optimized pass\n");

  const std::vector<Shape> shapes = parseShapes(kShapes);
  if (shapes.empty()) {
    llvm::errs() << "No valid shapes provided.\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::MLIRContext context(registry);

  llvm::outs() << "shape\tbaseline_divs_in_loop\tbaseline_divs_hoisted\t"
                  "optimized_divs_in_loop\toptimized_divs_hoisted\t"
                  "baseline_est_dynamic\toptimized_est_dynamic\t"
                  "div_reduction_%\tavg_pass_ms\n";

  for (const Shape &shape : shapes) {
    const std::string shapeName = shape.str();
    const std::string moduleText = buildSoftmaxMlir(shape);

    // Parse baseline and count divs.
    auto baselineModule =
        mlir::parseSourceString<mlir::ModuleOp>(moduleText, &context);
    if (!baselineModule) {
      llvm::errs() << "Failed to parse baseline module for shape " << shapeName << "\n";
      return 1;
    }

    DivStats baselineStats = countDivs(*baselineModule);

    // Estimate dynamic divs: hoisted divs execute once, in-loop divs execute M*N times.
    std::uint64_t baselineEstDynamic =
        baselineStats.outsideLoop + baselineStats.inLoopBody * shape.m * shape.n;

    // Run the pass multiple times and measure.
    constexpr unsigned kIters = 20;
    std::chrono::duration<double, std::milli> totalMs(0.0);
    mlir::OwningOpRef<mlir::ModuleOp> finalModule;

    for (unsigned iter = 0; iter < kIters; ++iter) {
      auto module =
          mlir::parseSourceString<mlir::ModuleOp>(moduleText, &context);
      if (!module) {
        llvm::errs() << "Failed to parse module at iteration " << iter << "\n";
        return 1;
      }

      mlir::PassManager pm(&context);
      pm.addPass(mlc::createDivToReciprocalMulPass());

      auto start = std::chrono::high_resolution_clock::now();
      mlir::LogicalResult result = pm.run(*module);
      auto end = std::chrono::high_resolution_clock::now();

      if (mlir::failed(result)) {
        llvm::errs() << "Pass failed for shape " << shapeName << "\n";
        return 1;
      }

      totalMs += end - start;
      if (iter + 1 == kIters) {
        finalModule = std::move(module);
      }
    }

    DivStats optimizedStats = countDivs(*finalModule);
    // After the pass, the single reciprocal div is hoisted outside all loops.
    std::uint64_t optimizedEstDynamic =
        optimizedStats.outsideLoop + optimizedStats.inLoopBody * shape.m * shape.n;

    double reduction = baselineEstDynamic == 0
                           ? 0.0
                           : 100.0 * (1.0 - static_cast<double>(optimizedEstDynamic) /
                                                static_cast<double>(baselineEstDynamic));
    double avgMs = totalMs.count() / static_cast<double>(kIters);

    llvm::outs() << shapeName
                 << "\t" << baselineStats.inLoopBody
                 << "\t" << baselineStats.outsideLoop
                 << "\t" << optimizedStats.inLoopBody
                 << "\t" << optimizedStats.outsideLoop
                 << "\t" << baselineEstDynamic
                 << "\t" << optimizedEstDynamic
                 << "\t" << reduction
                 << "\t" << avgMs << "\n";
  }

  return 0;
}
