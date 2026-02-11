#ifndef MLC_COMPILER_PIPELINE_LOWERINGPIPELINE_H
#define MLC_COMPILER_PIPELINE_LOWERINGPIPELINE_H

#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace mlir {
class MLIRContext;
}

namespace mlc {

enum class PipelineMode {
  kBaseline,
  kOptimized,
};

struct PipelineConfig {
  std::string inputPath;
  std::string outputDir;
  std::string llcPath;
  PipelineMode mode = PipelineMode::kOptimized;
};

struct PipelineArtifacts {
  std::string stage0InputMlir;
  std::string stage1OptimizedMlir;
  std::string stage2LlvmDialectMlir;
  std::string stage3LlvmIr;
  std::string stage4Ptx;
};

mlir::LogicalResult runLoweringPipeline(mlir::MLIRContext &context,
                                        const PipelineConfig &config,
                                        PipelineArtifacts &artifacts,
                                        llvm::raw_ostream &log);

}  // namespace mlc

#endif  // MLC_COMPILER_PIPELINE_LOWERINGPIPELINE_H
