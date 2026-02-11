add_library(MLCRuntime STATIC
  CudaRuntime.cpp
)

target_include_directories(MLCRuntime
  PUBLIC
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(MLCRuntime
  PUBLIC
    ${CMAKE_DL_LIBS}
)
