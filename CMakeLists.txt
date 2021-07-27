cmake_minimum_required(VERSION 3.19)

project(ParallelComputing_ParticleSimulation)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory("${PROJECT_SOURCE_DIR}/Serial" "${PROJECT_SOURCE_DIR}/Serial/Serial_out")
add_subdirectory("${PROJECT_SOURCE_DIR}/MPI" "${PROJECT_SOURCE_DIR}/MPI/MPI_out")
add_subdirectory("${PROJECT_SOURCE_DIR}/OpenMP" "${PROJECT_SOURCE_DIR}/OpenMP/OpenMP_out")
add_subdirectory("${PROJECT_SOURCE_DIR}/CUDA" "${PROJECT_SOURCE_DIR}/CUDA/CUDA_out")