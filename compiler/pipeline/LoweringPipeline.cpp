#include "compiler/pipeline/LoweringPipeline.h"

#include "mlc/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#if __has_include("mlir/Parser/Parser.h")
#include "mlir/Parser/Parser.h"
#else
#include "mlir/Parser.h"
#endif
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/ADT/Triple.h"

#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace mlc {
namespace {

static mlir::OwningOpRef<mlir::ModuleOp> parseModuleFromFile(
    mlir::MLIRContext &context, llvm::StringRef inputPath, llvm::raw_ostream &log) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputPath);
  if (!fileOrErr) {
    log << "Failed to read input MLIR file: " << inputPath << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

static mlir::LogicalResult writeTextFile(llvm::StringRef outputPath,
                                         llvm::StringRef content,
                                         llvm::raw_ostream &log) {
  std::error_code ec;
  llvm::raw_fd_ostream out(outputPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    log << "Failed to write file '" << outputPath << "': " << ec.message() << "\n";
    return mlir::failure();
  }
  out << content;
  return mlir::success();
}

static mlir::LogicalResult writeModuleToFile(mlir::ModuleOp module,
                                             llvm::StringRef outputPath,
                                             llvm::raw_ostream &log) {
  std::string text;
  llvm::raw_string_ostream os(text);
  module->print(os);
  os.flush();
  return writeTextFile(outputPath, text, log);
}

static mlir::LogicalResult runOptStage(mlir::MLIRContext &context,
                                       mlir::ModuleOp module,
                                       PipelineMode mode,
                                       llvm::raw_ostream &log) {
  mlir::PassManager pm(&context);
  if (mode == PipelineMode::kOptimized) {
    pm.addPass(createDivToReciprocalMulPass());
  }

  if (mlir::failed(pm.run(module))) {
    log << "Optimization stage failed.\n";
    return mlir::failure();
  }
  return mlir::success();
}

static mlir::LogicalResult runLowerToLlvmDialectStage(mlir::MLIRContext &context,
                                                      mlir::ModuleOp module,
                                                      llvm::raw_ostream &log) {
  mlir::PassManager pm(&context);
  // Keep this sequence stable for LLVM/MLIR 15: func conversion must run
  // before cf/arith conversion to avoid SCF/CF block argument type mismatches.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::cf::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::arith::createConvertArithmeticToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(module))) {
    log << "Lowering stage to LLVM dialect failed.\n";
    return mlir::failure();
  }
  return mlir::success();
}

static mlir::LogicalResult injectKernelWrapper(llvm::Module &llvmModule,
                                               llvm::raw_ostream &log) {
  llvm::Function *softmaxFn = llvmModule.getFunction("softmax_norm");
  if (!softmaxFn) {
    log << "Expected function 'softmax_norm' in LLVM IR module.\n";
    return mlir::failure();
  }

  if (softmaxFn->arg_size() != 1 || !softmaxFn->getReturnType()->isFloatTy() ||
      !softmaxFn->getArg(0)->getType()->isFloatTy()) {
    log << "Function 'softmax_norm' must have signature float(float).\n";
    return mlir::failure();
  }

  if (llvm::Function *existing = llvmModule.getFunction("softmax_kernel")) {
    existing->eraseFromParent();
  }

  llvm::LLVMContext &ctx = llvmModule.getContext();
  llvm::Type *f32Ty = llvm::Type::getFloatTy(ctx);
  llvm::Type *ptrTy = llvm::PointerType::get(ctx, 0);
  auto *wrapperTy = llvm::FunctionType::get(
      llvm::Type::getVoidTy(ctx), {ptrTy, f32Ty}, false);

  llvm::Function *wrapper = llvm::Function::Create(
      wrapperTy, llvm::GlobalValue::ExternalLinkage, "softmax_kernel", llvmModule);
  wrapper->setCallingConv(llvm::CallingConv::PTX_Kernel);

  auto argIt = wrapper->arg_begin();
  llvm::Argument *outPtr = &*argIt++;
  outPtr->setName("out");
  llvm::Argument *sum = &*argIt;
  sum->setName("sum");

  llvm::IRBuilder<> builder(llvm::BasicBlock::Create(ctx, "entry", wrapper));
  llvm::Value *value = builder.CreateCall(softmaxFn, {sum});
  builder.CreateStore(value, outPtr);
  builder.CreateRetVoid();

  llvm::NamedMDNode *annotations = llvmModule.getOrInsertNamedMetadata("nvvm.annotations");
  llvm::Metadata *mdValues[] = {
      llvm::ValueAsMetadata::get(wrapper),
      llvm::MDString::get(ctx, "kernel"),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
          llvm::Type::getInt32Ty(ctx), 1)),
  };
  annotations->addOperand(llvm::MDNode::get(ctx, mdValues));

  return mlir::success();
}

static mlir::LogicalResult writeLlvmIrFile(mlir::ModuleOp module,
                                           llvm::StringRef outputPath,
                                           llvm::raw_ostream &log) {
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    log << "Failed to translate MLIR LLVM dialect module to LLVM IR.\n";
    return mlir::failure();
  }

  llvmModule->setTargetTriple("nvptx64-nvidia-cuda");
  // PTX JIT under WSL rejects debug sections for older PTX ISA versions.
  llvm::StripDebugInfo(*llvmModule);

  if (mlir::failed(injectKernelWrapper(*llvmModule, log))) {
    return mlir::failure();
  }

  std::string llvmIr;
  llvm::raw_string_ostream os(llvmIr);
  llvmModule->print(os, nullptr);
  os.flush();

  return writeTextFile(outputPath, llvmIr, log);
}

static std::string resolveLlcPath(llvm::StringRef explicitPath) {
  if (!explicitPath.empty()) {
    return explicitPath.str();
  }

  llvm::ErrorOr<std::string> llcFromPath = llvm::sys::findProgramByName("llc");
  if (llcFromPath) {
    return *llcFromPath;
  }

  llvm::ErrorOr<std::string> llc15FromPath = llvm::sys::findProgramByName("llc-15");
  if (llc15FromPath) {
    return *llc15FromPath;
  }

  const char *homebrewLLC = "/opt/homebrew/opt/llvm/bin/llc";
  if (llvm::sys::fs::exists(homebrewLLC)) {
    return std::string(homebrewLLC);
  }

  const char *llvm15Llc = "/usr/lib/llvm-15/bin/llc";
  if (llvm::sys::fs::exists(llvm15Llc)) {
    return std::string(llvm15Llc);
  }

  return std::string();
}

static mlir::LogicalResult emitPtxWithLlc(llvm::StringRef llcPath,
                                          llvm::StringRef llvmIrPath,
                                          llvm::StringRef ptxPath,
                                          llvm::raw_ostream &log) {
  if (llcPath.empty()) {
    log << "Could not find llc executable.\n";
    return mlir::failure();
  }

  std::vector<std::string> storage = {
      llcPath.str(),
      "-mtriple=nvptx64-nvidia-cuda",
      "-mcpu=sm_80",
      "-filetype=asm",
      llvmIrPath.str(),
      "-o",
      ptxPath.str(),
  };

  llvm::SmallVector<llvm::StringRef, 8> argv;
  argv.reserve(storage.size());
  for (const std::string &arg : storage) {
    argv.push_back(arg);
  }

  std::string errMsg;
  bool executionFailed = false;
  int rc = llvm::sys::ExecuteAndWait(llcPath, argv, llvm::None, {}, 0, 0,
                                     &errMsg, &executionFailed);
  if (executionFailed || rc != 0) {
    log << "llc failed while emitting PTX (exit=" << rc << ").\n";
    if (!errMsg.empty()) {
      log << "llc error: " << errMsg << "\n";
    }
    return mlir::failure();
  }

  return mlir::success();
}

}  // namespace

mlir::LogicalResult runLoweringPipeline(mlir::MLIRContext &context,
                                        const PipelineConfig &config,
                                        PipelineArtifacts &artifacts,
                                        llvm::raw_ostream &log) {
  if (config.inputPath.empty() || config.outputDir.empty()) {
    log << "Both --input and --output-dir are required.\n";
    return mlir::failure();
  }

  if (std::error_code ec = llvm::sys::fs::create_directories(config.outputDir)) {
    log << "Failed to create output directory '" << config.outputDir
        << "': " << ec.message() << "\n";
    return mlir::failure();
  }

  artifacts.stage0InputMlir =
      (llvm::Twine(config.outputDir) + "/stage0_input.mlir").str();
  artifacts.stage1OptimizedMlir =
      (llvm::Twine(config.outputDir) + "/stage1_optimized.mlir").str();
  artifacts.stage2LlvmDialectMlir =
      (llvm::Twine(config.outputDir) + "/stage2_llvm_dialect.mlir").str();
  artifacts.stage3LlvmIr =
      (llvm::Twine(config.outputDir) + "/stage3_llvm_ir.ll").str();
  artifacts.stage4Ptx =
      (llvm::Twine(config.outputDir) + "/stage4_kernel.ptx").str();

  auto module = parseModuleFromFile(context, config.inputPath, log);
  if (!module) {
    log << "MLIR parse failed for input: " << config.inputPath << "\n";
    return mlir::failure();
  }

  if (mlir::failed(writeModuleToFile(*module, artifacts.stage0InputMlir, log))) {
    return mlir::failure();
  }

  if (mlir::failed(runOptStage(context, *module, config.mode, log))) {
    return mlir::failure();
  }
  if (mlir::failed(writeModuleToFile(*module, artifacts.stage1OptimizedMlir, log))) {
    return mlir::failure();
  }

  if (mlir::failed(runLowerToLlvmDialectStage(context, *module, log))) {
    return mlir::failure();
  }
  if (mlir::failed(writeModuleToFile(*module, artifacts.stage2LlvmDialectMlir, log))) {
    return mlir::failure();
  }

  if (mlir::failed(writeLlvmIrFile(*module, artifacts.stage3LlvmIr, log))) {
    return mlir::failure();
  }

  std::string llcPath = resolveLlcPath(config.llcPath);
  if (mlir::failed(
          emitPtxWithLlc(llcPath, artifacts.stage3LlvmIr, artifacts.stage4Ptx, log))) {
    return mlir::failure();
  }

  return mlir::success();
}

}  // namespace mlc
