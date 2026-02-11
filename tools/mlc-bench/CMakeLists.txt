add_executable(mlc-bench-pass
  mlc-bench.cpp
)

target_include_directories(mlc-bench-pass
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mlc-bench-pass
  PRIVATE
    MLCSoftmaxPasses
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRSupport
)
