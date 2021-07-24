cmake_minimum_required(VERSION 3.19)

project(OpenMP)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(OpenMP main.cpp common.h openmp.cpp)