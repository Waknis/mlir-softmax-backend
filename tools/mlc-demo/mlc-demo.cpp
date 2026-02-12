#include "compiler/pipeline/LoweringPipeline.h"
#include "runtime/CudaRuntime.h"

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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <string>

namespace {

llvm::cl::opt<std::string> kInput(
    "input", llvm::cl::desc("Input MLIR file"),
    llvm::cl::init("examples/softmax.mlir"));

llvm::cl::opt<std::string> kOutputDir(
    "output-dir", llvm::cl::desc("Directory for pipeline artifacts"),
    llvm::cl::init("build/demo"));

llvm::cl::opt<std::string> kMode(
    "mode", llvm::cl::desc("Pipeline mode: baseline or optimized"),
    llvm::cl::init("optimized"));

llvm::cl::opt<float> kSum(
    "sum", llvm::cl::desc("Softmax denominator input used by demo kernel"),
    llvm::cl::init(4.0f));

llvm::cl::opt<bool> kVerify(
    "verify", llvm::cl::desc("Verify against reference"), llvm::cl::init(false));

llvm::cl::opt<std::string> kLlc(
    "llc", llvm::cl::desc("Optional llc path override"), llvm::cl::init(""));

float referenceSoftmaxValue(float sum) {
  constexpr float kRows = 1024.0f;
  return kRows * (1.0f / sum);
}

}  // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "MLC demo: emit PTX from MLIR and run softmax kernel on CUDA driver\n");

  mlc::PipelineMode mode = mlc::PipelineMode::kOptimized;
  if (kMode == "baseline") {
    mode = mlc::PipelineMode::kBaseline;
  } else if (kMode != "optimized") {
    llvm::errs() << "Invalid --mode value. Use baseline or optimized.\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::MLIRContext context(registry);

  mlc::PipelineConfig config;
  config.inputPath = kInput;
  config.outputDir = kOutputDir;
  config.llcPath = kLlc;
  config.mode = mode;

  mlc::PipelineArtifacts artifacts;
  if (mlir::failed(mlc::runLoweringPipeline(context, config, artifacts,
                                             llvm::errs()))) {
    return 1;
  }

  auto ptxOrErr = llvm::MemoryBuffer::getFile(artifacts.stage4Ptx);
  if (!ptxOrErr) {
    llvm::errs() << "Failed to read generated PTX: " << artifacts.stage4Ptx << "\n";
    return 1;
  }

  std::string availabilityReason;
  if (!mlc::CudaRuntime::isCudaAvailable(availabilityReason)) {
    llvm::outs() << "SKIP: CUDA runtime unavailable: " << availabilityReason << "\n";
    return 0;
  }

  mlc::CudaRuntime runtime;
  std::string error;
  if (!runtime.initialize(error)) {
    llvm::errs() << "CUDA initialization failed: " << error << "\n";
    return 1;
  }

  if (!runtime.loadModuleFromPtx(ptxOrErr.get()->getBuffer().str(), error)) {
    llvm::errs() << "Failed to load PTX: " << error << "\n";
    return 1;
  }

  float gpuOutput = 0.0f;
  if (!runtime.launchSoftmaxKernel(kSum, gpuOutput, error)) {
    llvm::errs() << "Kernel launch failed: " << error << "\n";
    return 1;
  }

  float ref = referenceSoftmaxValue(kSum);
  float absErr = std::fabs(gpuOutput - ref);

  llvm::outs() << "sum=" << kSum << " gpu=" << gpuOutput << " ref=" << ref
               << " abs_err=" << absErr << "\n";

  if (kVerify && absErr > 1e-3f) {
    llvm::errs() << "Verification failed: abs_err=" << absErr << "\n";
    return 1;
  }

  llvm::outs() << "Demo completed. PTX: " << artifacts.stage4Ptx << "\n";
  return 0;
}
