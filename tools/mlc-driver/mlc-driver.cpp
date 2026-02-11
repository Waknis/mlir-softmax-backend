#include "compiler/pipeline/LoweringPipeline.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

llvm::cl::opt<std::string> kInput(
    "input", llvm::cl::desc("Input MLIR file"), llvm::cl::Required);

llvm::cl::opt<std::string> kOutputDir(
    "output-dir", llvm::cl::desc("Directory for pipeline artifacts"),
    llvm::cl::init("artifacts"));

llvm::cl::opt<std::string> kMode(
    "mode", llvm::cl::desc("Pipeline mode: baseline or optimized"),
    llvm::cl::init("optimized"));

llvm::cl::opt<std::string> kLlc(
    "llc", llvm::cl::desc("Optional llc path override"), llvm::cl::init(""));

}  // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "MLC driver: input MLIR -> optimized MLIR -> LLVM IR -> PTX\n");

  mlc::PipelineMode mode = mlc::PipelineMode::kOptimized;
  if (kMode == "baseline") {
    mode = mlc::PipelineMode::kBaseline;
  } else if (kMode != "optimized") {
    llvm::errs() << "Invalid --mode value. Use baseline or optimized.\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
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

  llvm::outs() << "Pipeline completed successfully.\n";
  llvm::outs() << "  input-mlir: " << artifacts.stage0InputMlir << "\n";
  llvm::outs() << "  optimized-mlir: " << artifacts.stage1OptimizedMlir << "\n";
  llvm::outs() << "  llvm-dialect-mlir: " << artifacts.stage2LlvmDialectMlir
               << "\n";
  llvm::outs() << "  llvm-ir: " << artifacts.stage3LlvmIr << "\n";
  llvm::outs() << "  ptx: " << artifacts.stage4Ptx << "\n";

  return 0;
}
