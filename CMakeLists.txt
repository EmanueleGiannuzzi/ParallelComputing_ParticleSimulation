cmake_minimum_required(VERSION 3.16)

project(ParallelComputing_ParticleSimulation)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory("${PROJECT_SOURCE_DIR}/Serial" "${PROJECT_SOURCE_DIR}/Serial/Serial_out")
add_subdirectory("${PROJECT_SOURCE_DIR}/MPI" "${PROJECT_SOURCE_DIR}/MPI/MPI_out")