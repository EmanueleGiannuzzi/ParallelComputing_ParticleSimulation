# cmake_minimum_required(VERSION <specify CMake version here>)
project(Serial)

set(CMAKE_CXX_STANDARD 14)

add_executable(Serial main.cpp common.h serial.cpp)