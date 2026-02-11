# CMake generated Testfile for 
# Source directory: /Users/mihirwaknis/Documents/GitHub/projectnamehere/test
# Build directory: /Users/mihirwaknis/Documents/GitHub/projectnamehere/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[mlc-div-to-reciprocal-mul]=] "/Users/mihirwaknis/miniforge3/lib/python3.12/site-packages/cmake/data/bin/cmake" "-DMLC_OPT=/Users/mihirwaknis/Documents/GitHub/projectnamehere/build/tools/mlc-opt/mlc-opt" "-DFILECHECK_EXE=/opt/homebrew/Cellar/llvm/21.1.8_1/bin/FileCheck" "-DINPUT_FILE=/Users/mihirwaknis/Documents/GitHub/projectnamehere/test/FileCheck/div_to_reciprocal_mul.mlir" "-DTMP_FILE=/Users/mihirwaknis/Documents/GitHub/projectnamehere/build/test/div_to_reciprocal_mul.out.mlir" "-P" "/Users/mihirwaknis/Documents/GitHub/projectnamehere/test/run_filecheck.cmake")
set_tests_properties([=[mlc-div-to-reciprocal-mul]=] PROPERTIES  _BACKTRACE_TRIPLES "/Users/mihirwaknis/Documents/GitHub/projectnamehere/test/CMakeLists.txt;12;add_test;/Users/mihirwaknis/Documents/GitHub/projectnamehere/test/CMakeLists.txt;0;")
