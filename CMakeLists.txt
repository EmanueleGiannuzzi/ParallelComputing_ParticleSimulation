# cmake_minimum_required(VERSION <specify CMake version here>)
project(ParallelComputing_ParticleSimulation)

set(CMAKE_CXX_STANDARD 14)

add_executable(ParallelComputing_ParticleSimulation main.cpp common.h serial.cpp)