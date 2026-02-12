#include "compiler/pipeline/LoweringPipeline.h"

#if __has_include("mlir/Dialect/Arith/IR/Arith.h")
#include "mlir/Dialect/Arith/IR/Arith.h"
#else
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
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

bool writeTextFile(llvm::StringRef path, llvm::StringRef text) {
  std::ofstream out(path.str(), std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    llvm::errs() << "Failed to open file for writing: " << path << "\n";
    return false;
  }
  out << text.str();
  return true;
}

bool runPipeline(mlir::MLIRContext &context,
                 const std::string &inputPath,
                 const std::string &outputDir,
                 mlc::PipelineMode mode,
                 const std::string &llcPath,
                 double &elapsedMs) {
  mlc::PipelineConfig config;
  config.inputPath = inputPath;
  config.outputDir = outputDir;
  config.mode = mode;
  config.llcPath = llcPath;

  mlc::PipelineArtifacts artifacts;

  auto start = std::chrono::steady_clock::now();
  mlir::LogicalResult result =
      mlc::runLoweringPipeline(context, config, artifacts, llvm::errs());
  auto end = std::chrono::steady_clock::now();

  elapsedMs =
      std::chrono::duration<double, std::milli>(end - start).count();
  return mlir::succeeded(result);
}

}  // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Softmax benchmark: timing table across shapes for baseline vs optimized pipeline\n");

  const std::vector<Shape> shapes = parseShapes(kShapes);
  if (shapes.empty()) {
    llvm::errs() << "No valid shapes provided.\n";
    return 1;
  }

  if (std::error_code ec = llvm::sys::fs::create_directories(kOutputRoot)) {
    llvm::errs() << "Failed to create output root: " << ec.message() << "\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::MLIRContext context(registry);

  llvm::outs() << "shape\tbaseline_ms\toptimized_ms\tspeedup\tbaseline_est_divs\t"
                  "optimized_est_divs\tdiv_reduction_%\n";

  for (const Shape &shape : shapes) {
    const std::string shapeName = shape.str();
    const std::string inputPath =
        (llvm::Twine(kOutputRoot) + "/" + shapeName + ".mlir").str();
    const std::string baselineOut =
        (llvm::Twine(kOutputRoot) + "/" + shapeName + "_baseline").str();
    const std::string optimizedOut =
        (llvm::Twine(kOutputRoot) + "/" + shapeName + "_optimized").str();

    if (!writeTextFile(inputPath, buildSoftmaxMlir(shape))) {
      return 1;
    }

    double baselineMs = 0.0;
    if (!runPipeline(context, inputPath, baselineOut, mlc::PipelineMode::kBaseline,
                     kLlc, baselineMs)) {
      llvm::errs() << "Baseline pipeline failed for shape " << shapeName << "\n";
      return 1;
    }

    double optimizedMs = 0.0;
    if (!runPipeline(context, inputPath, optimizedOut, mlc::PipelineMode::kOptimized,
                     kLlc, optimizedMs)) {
      llvm::errs() << "Optimized pipeline failed for shape " << shapeName << "\n";
      return 1;
    }

    const std::int64_t baselineDivs = shape.m * shape.n;
    const std::int64_t optimizedDivs = shape.m;
    const double divReduction =
        100.0 * (1.0 - static_cast<double>(optimizedDivs) /
                           static_cast<double>(baselineDivs));
    const double speedup = optimizedMs > 0.0 ? baselineMs / optimizedMs : 0.0;

    llvm::outs() << shapeName << "\t" << baselineMs << "\t" << optimizedMs
                 << "\t" << speedup << "\t" << baselineDivs << "\t"
                 << optimizedDivs << "\t" << divReduction << "\n";
  }

  return 0;
}
