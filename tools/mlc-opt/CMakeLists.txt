add_executable(mlc-opt
  mlc-opt.cpp
)

target_include_directories(mlc-opt
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mlc-opt
  PRIVATE
    MLCSoftmaxPasses
    MLIRArithmeticDialect
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIROptLib
    MLIRSCFDialect
)
