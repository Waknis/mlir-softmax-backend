add_executable(mlc-demo
  mlc-demo.cpp
  ${PROJECT_SOURCE_DIR}/compiler/pipeline/LoweringPipeline.cpp
)

target_include_directories(mlc-demo
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mlc-demo
  PRIVATE
    MLCSoftmaxPasses
    MLCRuntime
    MLIROptLib
    MLIRArithmeticToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRTargetLLVMIRExport
    MLIRToLLVMIRTranslationRegistration
    MLIRLLVMToLLVMIRTranslation
    MLIRParser
    MLIRSupport
)
