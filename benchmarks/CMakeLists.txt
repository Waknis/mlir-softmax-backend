# mlc-pass-analysis: measures *compile-time* effect of the div->recip*mul pass
# (pipeline wall time, static div-op counts) on synthetic softmax-shaped
# loops. NOT a runtime benchmark -- for runtime throughput on GPU, use
# `benchmarks/softmax_gpu_bench.py`.
add_executable(mlc-pass-analysis
  pass_static_analysis.cpp
  ${PROJECT_SOURCE_DIR}/compiler/pipeline/LoweringPipeline.cpp
)

target_include_directories(mlc-pass-analysis
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mlc-pass-analysis
  PRIVATE
    MLCSoftmaxPasses
    MLIROptLib
    MLIRArithmeticToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRLLVMToLLVMIRTranslation
    MLIRParser
    MLIRSupport
)

set_target_properties(mlc-pass-analysis PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
