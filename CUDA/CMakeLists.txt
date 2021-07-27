 
cmake_minimum_required(VERSION 3.19)

#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#    set(CMAKE_CUDA_ARCHITECTURES 75)
#endif()

project(CUDA LANGUAGES CUDA)

add_executable(CUDA main.cu gpu.cu common.h)
#add_executable(OPT main.cu opt.cu common.h)