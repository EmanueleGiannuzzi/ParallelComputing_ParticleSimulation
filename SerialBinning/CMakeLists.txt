cmake_minimum_required(VERSION 3.19)

project(SerialBinning)

set(CMAKE_CXX_STANDARD 14)

add_executable(SerialBinning main.cpp common.h serial_binning.cpp)