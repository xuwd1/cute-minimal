cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_COMPILER /usr/bin/g++)
project(cmake-minimal-gcc LANGUAGES CXX)

#standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#cutlass include
set(CUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
message(STATUS "CUTLASS_INCLUDE_DIR: ${CUTLASS_INCLUDE_DIR}")

#executable
add_executable(tester tester.cpp)
target_include_directories(tester PRIVATE ${CUTLASS_INCLUDE_DIR})
