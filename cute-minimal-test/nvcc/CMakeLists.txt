cmake_minimum_required(VERSION 3.23)
project(cmake-minimal-nvcc LANGUAGES CUDA)

#standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#cutlass include
set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
message(STATUS "CUTLASS_INCLUDE_DIR: ${CUTLASS_INCLUDE_DIR}")

#executable
add_executable(tester tester.cu)
target_include_directories(tester PRIVATE ${CUTLASS_INCLUDE_DIR})
