#ifndef MLC_PASSES_H
#define MLC_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlc {

std::unique_ptr<mlir::Pass> createDivToReciprocalMulPass();
void registerMLCPasses();

}  // namespace mlc

#endif  // MLC_PASSES_H
