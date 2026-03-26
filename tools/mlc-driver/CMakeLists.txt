add_executable(mlc-driver
  mlc-driver.cpp
  ${PROJECT_SOURCE_DIR}/compiler/pipeline/LoweringPipeline.cpp
)

target_include_directories(mlc-driver
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mlc-driver
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
