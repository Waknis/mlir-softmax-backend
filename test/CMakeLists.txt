find_program(
  FILECHECK_EXE
  NAMES FileCheck
  HINTS ${LLVM_TOOLS_BINARY_DIR} ${LLVM_BINARY_DIR}
  PATH_SUFFIXES bin
)

if(NOT FILECHECK_EXE)
  message(FATAL_ERROR "FileCheck not found. Install LLVM tools or set PATH.")
endif()

find_program(
  BASH_EXE
  NAMES bash
)

if(NOT BASH_EXE)
  message(FATAL_ERROR "bash not found.")
endif()

add_test(
  NAME mlc-div-to-reciprocal-mul
  COMMAND ${CMAKE_COMMAND}
    -DMLC_OPT=$<TARGET_FILE:mlc-opt>
    -DFILECHECK_EXE=${FILECHECK_EXE}
    -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/FileCheck/div_to_reciprocal_mul.mlir
    -DTMP_FILE=${CMAKE_CURRENT_BINARY_DIR}/div_to_reciprocal_mul.out.mlir
    -P ${CMAKE_CURRENT_SOURCE_DIR}/run_filecheck.cmake
)

add_test(
  NAME mlc-driver-end-to-end
  COMMAND ${BASH_EXE}
    ${PROJECT_SOURCE_DIR}/tests/test_pipeline_end_to_end.sh
    $<TARGET_FILE:mlc-driver>
    ${PROJECT_SOURCE_DIR}/examples/softmax.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/pipeline_e2e
)

add_executable(mlc-softmax-gpu-correctness
  ${PROJECT_SOURCE_DIR}/tests/test_softmax_gpu_correctness.cpp
)

add_test(
  NAME mlc-softmax-gpu-correctness
  COMMAND $<TARGET_FILE:mlc-softmax-gpu-correctness>
    $<TARGET_FILE:mlc-demo>
    ${PROJECT_SOURCE_DIR}/examples/softmax.mlir
    ${CMAKE_CURRENT_BINARY_DIR}/gpu_correctness
)

add_test(
  NAME mlc-benchmark-10-shapes
  COMMAND ${BASH_EXE}
    ${PROJECT_SOURCE_DIR}/tests/test_benchmark_shapes.sh
    $<TARGET_FILE:softmax-benchmark>
)
