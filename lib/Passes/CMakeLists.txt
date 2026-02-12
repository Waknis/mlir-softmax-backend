add_library(MLCSoftmaxPasses STATIC
  DivToReciprocalMulPass.cpp
)

target_include_directories(MLCSoftmaxPasses
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(MLCSoftmaxPasses
  PUBLIC
    MLIRArithmeticDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRSCFDialect
)
