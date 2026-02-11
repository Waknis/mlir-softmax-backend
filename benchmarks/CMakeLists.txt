add_executable(softmax-benchmark
  softmax_benchmark.cpp
  ${PROJECT_SOURCE_DIR}/compiler/pipeline/LoweringPipeline.cpp
)

target_include_directories(softmax-benchmark
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(softmax-benchmark
  PRIVATE
    MLCSoftmaxPasses
    MLIROptLib
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRIndexToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRLLVMToLLVMIRTranslation
    MLIRBuiltinToLLVMIRTranslation
    MLIRParser
    MLIRSupport
)

set_target_properties(softmax-benchmark PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
