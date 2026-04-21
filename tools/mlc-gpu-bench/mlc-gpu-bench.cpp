#include "compiler/pipeline/LoweringPipeline.h"
#include "runtime/CudaRuntime.h"

#if __has_include("mlir/Dialect/Arith/IR/Arith.h")
#include "mlir/Dialect/Arith/IR/Arith.h"
#else
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#endif
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct ModeSpec {
  mlc::PipelineMode mode;
  llvm::StringRef name;
};

llvm::cl::opt<std::string> kSizes(
    "sizes",
    llvm::cl::desc("Comma-separated 1-D vector sizes to compile and benchmark."),
    llvm::cl::init("1024,4096,16384,65536"));

llvm::cl::opt<unsigned> kWarmup(
    "warmup",
    llvm::cl::desc("Number of untimed warmup kernel launches per size/mode."),
    llvm::cl::init(25));

llvm::cl::opt<unsigned> kIters(
    "iters",
    llvm::cl::desc("Number of timed kernel launches per size/mode."),
    llvm::cl::init(100));

llvm::cl::opt<std::string> kMode(
    "mode",
    llvm::cl::desc("Benchmark mode: baseline, optimized, or both."),
    llvm::cl::init("both"));

llvm::cl::opt<std::string> kOutputRoot(
    "output-root",
    llvm::cl::desc("Directory where synthetic MLIR inputs and PTX artifacts are written."),
    llvm::cl::init("build/benchmark_runs/gpu"));

llvm::cl::opt<std::string> kLlc(
    "llc",
    llvm::cl::desc("Optional llc path override."),
    llvm::cl::init(""));

llvm::cl::opt<float> kSum(
    "sum",
    llvm::cl::desc("Scalar denominator used by the normalization kernel."),
    llvm::cl::init(4.0f));

llvm::cl::opt<bool> kCsv(
    "csv",
    llvm::cl::desc("Emit CSV instead of tab-separated output."),
    llvm::cl::init(false));

llvm::cl::opt<bool> kVerify(
    "verify",
    llvm::cl::desc("Fail if max absolute error exceeds 1e-5."),
    llvm::cl::init(false));

std::vector<std::int64_t> parseSizes() {
  llvm::SmallVector<llvm::StringRef> tokens;
  llvm::SplitString(llvm::StringRef(kSizes), tokens, ",");

  std::vector<std::int64_t> sizes;
  sizes.reserve(tokens.size());
  for (llvm::StringRef token : tokens) {
    token = token.trim();
    if (token.empty()) {
      continue;
    }

    std::int64_t value = 0;
    if (token.getAsInteger(10, value) || value <= 0) {
      llvm::errs() << "Invalid size token: " << token << "\n";
      continue;
    }
    sizes.push_back(value);
  }

  return sizes;
}

std::vector<ModeSpec> parseModes() {
  if (kMode == "baseline") {
    return {{mlc::PipelineMode::kBaseline, "baseline"}};
  }
  if (kMode == "optimized") {
    return {{mlc::PipelineMode::kOptimized, "optimized"}};
  }
  if (kMode == "both") {
    return {
        {mlc::PipelineMode::kBaseline, "baseline"},
        {mlc::PipelineMode::kOptimized, "optimized"},
    };
  }

  llvm::errs() << "Invalid --mode value. Use baseline, optimized, or both.\n";
  return {};
}

std::string buildSoftmaxMlir(std::int64_t n) {
  std::string nStr = std::to_string(n);
  std::string text;
  llvm::raw_string_ostream os(text);
  os << "module {\n"
     << "  func.func @softmax_norm(%input: memref<" << nStr << "xf32>,\n"
     << "                          %output: memref<" << nStr << "xf32>,\n"
     << "                          %sum: f32) {\n"
     << "    %c0 = arith.constant 0 : index\n"
     << "    %cN = arith.constant " << nStr << " : index\n"
     << "    %c1 = arith.constant 1 : index\n"
     << "    scf.for %i = %c0 to %cN step %c1 {\n"
     << "      %x = memref.load %input[%i] : memref<" << nStr << "xf32>\n"
     << "      %y = arith.divf %x, %sum : f32\n"
     << "      memref.store %y, %output[%i] : memref<" << nStr << "xf32>\n"
     << "    }\n"
     << "    return\n"
     << "  }\n"
     << "}\n";
  os.flush();
  return text;
}

mlir::LogicalResult writeTextFile(llvm::StringRef path, llvm::StringRef content) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "Failed to write file '" << path << "': " << ec.message()
                 << "\n";
    return mlir::failure();
  }
  out << content;
  return mlir::success();
}

double effectiveGiBsPerSecond(std::int64_t n, double avgKernelMs) {
  if (avgKernelMs <= 0.0) {
    return 0.0;
  }
  const double movedBytes =
      static_cast<double>(n) * static_cast<double>(sizeof(float) * 2);
  const double seconds = avgKernelMs / 1000.0;
  return movedBytes / seconds / (1024.0 * 1024.0 * 1024.0);
}

void printHeader() {
  if (kCsv) {
    llvm::outs()
        << "mode,size,warmup,iters,avg_kernel_ms,effective_gib_s,max_abs_err,"
           "speedup_vs_baseline\n";
    return;
  }

  llvm::outs()
      << "mode\tsize\twarmup\titers\tavg_kernel_ms\teffective_gib_s\t"
         "max_abs_err\tspeedup_vs_baseline\n";
}

void printRow(llvm::StringRef modeName,
              std::int64_t size,
              double avgKernelMs,
              double gibPerSecond,
              float maxAbsErr,
              double speedupVsBaseline) {
  if (kCsv) {
    llvm::outs() << modeName << "," << size << "," << kWarmup << "," << kIters
                 << "," << llvm::format("%.6f", avgKernelMs)
                 << "," << llvm::format("%.6f", gibPerSecond)
                 << "," << llvm::format("%.8f", maxAbsErr)
                 << "," << llvm::format("%.6f", speedupVsBaseline) << "\n";
    return;
  }

  llvm::outs() << modeName << "\t" << size << "\t" << kWarmup << "\t" << kIters
               << "\t" << llvm::format("%.6f", avgKernelMs)
               << "\t" << llvm::format("%.6f", gibPerSecond)
               << "\t" << llvm::format("%.8f", maxAbsErr)
               << "\t" << llvm::format("%.6f", speedupVsBaseline) << "\n";
}

}  // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "MLC GPU benchmark: compile synthetic normalization kernels and time generated PTX\n");

  if (kIters == 0) {
    llvm::errs() << "--iters must be greater than zero.\n";
    return 1;
  }

  const std::vector<std::int64_t> sizes = parseSizes();
  if (sizes.empty()) {
    llvm::errs() << "No valid sizes provided.\n";
    return 1;
  }

  const std::vector<ModeSpec> modes = parseModes();
  if (modes.empty()) {
    return 1;
  }

  std::string availabilityReason;
  if (!mlc::CudaRuntime::isCudaAvailable(availabilityReason)) {
    llvm::outs() << "SKIP: CUDA runtime unavailable: " << availabilityReason << "\n";
    return 0;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::MLIRContext context(registry);

  mlc::CudaRuntime runtime;
  std::string error;
  if (!runtime.initialize(error)) {
    llvm::errs() << "CUDA initialization failed: " << error << "\n";
    return 1;
  }

  printHeader();

  std::unordered_map<std::int64_t, double> baselineBySize;
  for (std::int64_t size : sizes) {
    const std::string sizeDir =
        (llvm::Twine(kOutputRoot) + "/n" + llvm::Twine(size)).str();
    if (std::error_code ec = llvm::sys::fs::create_directories(sizeDir)) {
      llvm::errs() << "Failed to create benchmark directory '" << sizeDir
                   << "': " << ec.message() << "\n";
      return 1;
    }

    const std::string inputPath = (llvm::Twine(sizeDir) + "/input.mlir").str();
    if (mlir::failed(writeTextFile(inputPath, buildSoftmaxMlir(size)))) {
      return 1;
    }

    std::vector<float> inputHost(static_cast<std::size_t>(size), 0.0f);
    for (std::int64_t i = 0; i < size; ++i) {
      inputHost[static_cast<std::size_t>(i)] =
          1.0f + static_cast<float>(i % 19) * 0.03125f;
    }
    std::vector<float> outputHost(static_cast<std::size_t>(size), 0.0f);

    for (const ModeSpec &modeSpec : modes) {
      const std::string outputDir =
          (llvm::Twine(sizeDir) + "/" + modeSpec.name).str();

      mlc::PipelineConfig config;
      config.inputPath = inputPath;
      config.outputDir = outputDir;
      config.llcPath = kLlc;
      config.mode = modeSpec.mode;

      mlc::PipelineArtifacts artifacts;
      if (mlir::failed(mlc::runLoweringPipeline(context, config, artifacts,
                                                llvm::errs()))) {
        return 1;
      }

      auto ptxOrErr = llvm::MemoryBuffer::getFile(artifacts.stage4Ptx);
      if (!ptxOrErr) {
        llvm::errs() << "Failed to read generated PTX: " << artifacts.stage4Ptx
                     << "\n";
        return 1;
      }

      if (!runtime.loadModuleFromPtx(ptxOrErr.get()->getBuffer().str(), error)) {
        llvm::errs() << "Failed to load PTX: " << error << "\n";
        return 1;
      }

      mlc::KernelBenchmarkConfig benchConfig;
      benchConfig.warmupIterations = kWarmup;
      benchConfig.timedIterations = kIters;

      mlc::KernelBenchmarkResult benchResult;
      if (!runtime.benchmarkSoftmaxMemrefKernel(inputHost.data(),
                                                outputHost.data(),
                                                size,
                                                kSum,
                                                benchConfig,
                                                benchResult,
                                                error)) {
        llvm::errs() << "Benchmark failed for size " << size << " in "
                     << modeSpec.name << " mode: " << error << "\n";
        return 1;
      }

      if (kVerify && benchResult.maxAbsError > 1e-5f) {
        llvm::errs() << "Verification failed for size " << size << " in "
                     << modeSpec.name << " mode: max_abs_err="
                     << benchResult.maxAbsError << "\n";
        return 1;
      }

      const double gibPerSecond =
          effectiveGiBsPerSecond(size, benchResult.avgKernelMs);
      double speedupVsBaseline = 1.0;
      if (modeSpec.mode == mlc::PipelineMode::kBaseline) {
        baselineBySize[size] = benchResult.avgKernelMs;
      } else {
        auto baselineIt = baselineBySize.find(size);
        if (baselineIt != baselineBySize.end() && benchResult.avgKernelMs > 0.0) {
          speedupVsBaseline = baselineIt->second / benchResult.avgKernelMs;
        }
      }

      printRow(modeSpec.name,
               size,
               benchResult.avgKernelMs,
               gibPerSecond,
               benchResult.maxAbsError,
               speedupVsBaseline);
    }
  }

  return 0;
}
