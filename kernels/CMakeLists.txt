# Builds the hand-written softmax CUDA kernels into a shared library that the
# Python benchmark loads via ctypes. CUDA is optional for the project as a
# whole (runtime/ loads libcuda.so via dlopen at runtime), so this target is
# skipped when nvcc is unavailable.

include(CheckLanguage)
check_language(CUDA)

if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "kernels/: CUDA compiler not found -- skipping hand kernel build")
    return()
endif()

enable_language(CUDA)

message(STATUS "kernels/: CUDA ${CMAKE_CUDA_COMPILER_VERSION} at ${CMAKE_CUDA_COMPILER}")

add_library(mlc_softmax_kernels SHARED
    softmax_online.cu
)

target_include_directories(mlc_softmax_kernels
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(mlc_softmax_kernels PROPERTIES
    # Target sm_80 (Ampere) as the floor; add 90 (Hopper) and 120 (Blackwell,
    # e.g. RTX 5060 Ti) for native codegen. Older architectures fall back to
    # PTX JIT via the embedded PTX from the lowest listed arch.
    CUDA_ARCHITECTURES "80;89;90;120"
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    OUTPUT_NAME "mlc_softmax_kernels"
)

target_compile_options(mlc_softmax_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
