cmake_minimum_required(VERSION 3.26)
project(vollab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Contraction off so fused multiply-add cannot silently change results between
# this engine and the NumPy reference.
add_compile_options(-ffp-contract=off)

find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core cpp/src/core.cpp)
target_include_directories(_core PRIVATE cpp/include)
install(TARGETS _core LIBRARY DESTINATION vollab)
