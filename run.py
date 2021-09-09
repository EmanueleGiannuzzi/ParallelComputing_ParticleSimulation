import subprocess
from typing import List
import numpy as np


def run_naive(number_of_particles: int,
              particle_initialization_seed: int,
              cutoff: float,
              rendering: bool) -> float:

    # store output of the program as a byte string in s
    s = subprocess.check_output("Naive/Naive_out/Naive -n {0} -s {1} -o Naive/Naive_out/naive.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    # decode s to a normal string
    output = s.decode("utf-8")
    print("Naive - " + output)

    if rendering is True:
        subprocess.check_call(
            "python rendering/render.py Naive/Naive_out/naive.parts.out rendering/naive_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def run_serial(number_of_particles: int,
               particle_initialization_seed: int,
               cutoff: float,
               correctness_check: bool,
               rendering: bool) -> float:

    s = subprocess.check_output("Serial/Serial_out/Serial -n {0} -s {1} -o Serial/Serial_out/serial.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    output = s.decode("utf-8")
    print("Serial - " + output)

    if correctness_check is True:
        subprocess.check_call(
            "python correctness-check/correctness-check.py Serial/Serial_out/serial.parts.out Naive/Naive_out/naive.parts.out",
            shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        subprocess.check_call(
            "python rendering/render.py Serial/Serial_out/serial.parts.out rendering/serial_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def run_openmp(number_of_particles: int,
               particle_initialization_seed: int,
               cutoff: float,
               correctness_check: bool,
               rendering: bool) -> float:

    s = subprocess.check_output("OpenMP/OpenMP_out/OpenMP -n {0} -s {1} -o OpenMP/OpenMP_out/openmp.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    output = s.decode("utf-8")
    print("OpenMP - " + output)

    if correctness_check is True:
        subprocess.check_call(
            "python correctness-check/correctness-check.py OpenMP/OpenMP_out/openmp.parts.out Naive/Naive_out/naive.parts.out",
            shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        subprocess.check_call(
            "python rendering/render.py OpenMP/OpenMP_out/openmp.parts.out rendering/openmp_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def run_mpi(number_of_processes: int,
            number_of_particles: int,
            particle_initialization_seed: int,
            cutoff: float,
            correctness_check: bool,
            rendering: bool) -> float:

    s = subprocess.check_output("mpirun -n {0} MPI/MPI_out/MPI -n {1} -s {2} -o MPI/MPI_out/mpi.parts.out"
                                .format(number_of_processes, number_of_particles, particle_initialization_seed),
                                shell=True)
    output = s.decode("utf-8")
    print("MPI - " + output)

    if correctness_check is True:
        subprocess.check_call(
            "python correctness-check/correctness-check.py MPI/MPI_out/mpi.parts.out Naive/Naive_out/naive.parts.out",
            shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        subprocess.check_call(
            "python rendering/render.py MPI/MPI_out/mpi.parts.out rendering/mpi_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def run_cuda(number_of_particles: int,
             particle_initialization_seed: int,
             cutoff: float,
             correctness_check: bool,
             rendering: bool) -> float:

    s = subprocess.check_output("CUDA/CUDA_out/CUDA -n {0} -s {1} -o CUDA/CUDA_out/cuda.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    output = s.decode("utf-8")
    print("CUDA - " + output)

    if correctness_check is True:
        subprocess.check_call(
            "python correctness-check/correctness-check.py CUDA/CUDA_out/cuda.parts.out Naive/Naive_out/naive.parts.out",
            shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        subprocess.check_call(
            "python rendering/render.py CUDA/CUDA_out/cuda.parts.out rendering/cuda_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def run_cuda_opt(number_of_particles: int,
                 particle_initialization_seed: int,
                 cutoff: float,
                 correctness_check: bool,
                 rendering: bool) -> float:

    s = subprocess.check_output("CUDA/CUDA_out/OPT -n {0} -s {1} -o CUDA/CUDA_out/opt.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    output = s.decode("utf-8")
    print("OPT - " + output)

    if correctness_check is True:
        subprocess.check_call(
            "python correctness-check/correctness-check.py CUDA/CUDA_out/opt.parts.out Naive/Naive_out/naive.parts.out",
            shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        subprocess.check_call(
            "python rendering/render.py CUDA/CUDA_out/opt.parts.out rendering/opt_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


if __name__ == "__main__":
    number_of_simulations: int = 4
    number_of_particles_per_simulation: List[int] = [2**j for j in range(10, 10 + number_of_simulations)]
    particle_initialization_seed: int = 42
    cutoff: float = 0.01
    max_number_of_processes: int = 6
    correctness_check: bool = False
    rendering: bool = False

    naive_times: List[float] = []
    serial_times: List[float] = []
    openmp_times: List[float] = []
    mpi_times: List[List[float]] = []
    cuda_times: List[float] = []
    cuda_opt_times: List[float] = []

    for number_of_particles in number_of_particles_per_simulation:
        if correctness_check is True:
            naive_times.append(run_naive(number_of_particles, particle_initialization_seed, cutoff, rendering))

        serial_times.append(run_serial(number_of_particles, particle_initialization_seed, cutoff,
                                       correctness_check, rendering))
        openmp_times.append(run_openmp(number_of_particles, particle_initialization_seed, cutoff,
                                       correctness_check, rendering))
        mpi_times.append([run_mpi(i, number_of_particles, particle_initialization_seed, cutoff,
                                  correctness_check, rendering)
                          for i in range(1, max_number_of_processes + 1)])
        cuda_times.append(run_cuda(number_of_particles, particle_initialization_seed, cutoff,
                                   correctness_check, rendering))
        cuda_opt_times.append(run_cuda_opt(number_of_particles, particle_initialization_seed, cutoff,
                                           correctness_check, rendering))

    np.savetxt("naive_times.csv", X=np.array(naive_times),
               header="1, 2, 3, 4, 5, 6", delimiter=",")
    np.savetxt("mpi_times.csv", X=np.array(mpi_times),
               header="1, 2, 3, 4, 5, 6", delimiter=",")
    timing_results = np.block([np.array(serial_times)[:, np.newaxis],
                               np.array(openmp_times)[:, np.newaxis],
                               np.array([min(times) for times in mpi_times])[:, np.newaxis],
                               np.array(cuda_times)[:, np.newaxis],
                               np.array(cuda_opt_times)[:, np.newaxis]])
    np.savetxt("timing_results.csv", X=timing_results,
               header="serial, openmp, mpi, cuda, cuda_opt", delimiter=",")
