cmake_minimum_required(VERSION 3.19)

project(MPI)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI)

include_directories(${MPI_INCLUDE_PATH})

add_executable(MPI main.cpp mpi.cpp common.h)

target_link_libraries(MPI ${MPI_LIBRARIES})