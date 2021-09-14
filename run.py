import subprocess
from typing import List
import numpy as np
import os


def run_naive(number_of_particles: int,
              particle_initialization_seed: int,
              cutoff: float,
              rendering: bool = False) -> float:

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
               correctness_check: bool = False,
               rendering: bool = False,
               heap_profiler: bool = False) -> float:

    c = "Serial/Serial_out/Serial -n {0} -s {1} -o Serial/Serial_out/serial.parts.out"
    s = subprocess.check_output(
        (c if not heap_profiler else "valgrind --tool=massif --massif-out-file=heap-profiler/serial/massif.out.{0} "
                                     + c).format(number_of_particles, particle_initialization_seed), shell=True)
    output = s.decode("utf-8")
    print("Serial - " + output)

    if correctness_check is True:
        c = "python correctness-check/correctness-check.py"
        subprocess.check_call(
            c + " Serial/Serial_out/serial.parts.out Naive/Naive_out/naive.parts.out", shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        c = "python rendering/render.py"
        subprocess.check_call(
            c + " Serial/Serial_out/serial.parts.out rendering/serial_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def run_openmp(number_of_threads: int,
               number_of_particles: int,
               particle_initialization_seed: int,
               cutoff: float,
               correctness_check: bool = False,
               rendering: bool = False,
               heap_profiler: bool = False) -> float:

    my_env = os.environ.copy()
    my_env['OMP_NUM_THREADS'] = str(number_of_threads)
    c = "OpenMP/OpenMP_out/OpenMP -n {0} -s {1} -o OpenMP/OpenMP_out/openmp.parts.out"
    s = subprocess.check_output(
        (c if not heap_profiler else "valgrind --tool=massif --massif-out-file=heap-profiler/openmp/massif.out.{0}_{2} "
                                     + c).format(number_of_particles, particle_initialization_seed, number_of_threads),
        shell=True, env=my_env)
    output = s.decode("utf-8")
    print("OpenMP with {0} threads - ".format(number_of_threads) + output)

    if correctness_check is True:
        c = "python correctness-check/correctness-check.py"
        subprocess.check_call(
            c + " OpenMP/OpenMP_out/openmp.parts.out Naive/Naive_out/naive.parts.out",
            shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        c = "python rendering/render.py"
        subprocess.check_call(
            c + " OpenMP/OpenMP_out/openmp.parts.out rendering/openmp_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def run_mpi(number_of_processes: int,
            number_of_particles: int,
            particle_initialization_seed: int,
            cutoff: float,
            correctness_check: bool = False,
            rendering: bool = False,
            heap_profiler: bool = False) -> float:

    c1 = "mpirun -n {0}"
    c2 = " MPI/MPI_out/MPI -n {1} -s {2} -o MPI/MPI_out/mpi.parts.out"
    s = subprocess.check_output(
        (c1 + c2 if not heap_profiler else
         c1 + " valgrind --tool=massif --massif-out-file=heap-profiler/mpi/massif.out.{1}_{0}" + c2)
        .format(number_of_processes, number_of_particles, particle_initialization_seed), shell=True)
    output = s.decode("utf-8")
    print("MPI with {0} processes - ".format(number_of_processes) + output)

    if correctness_check is True:
        c = "python correctness-check/correctness-check.py"
        subprocess.check_call(
            c + " MPI/MPI_out/mpi.parts.out Naive/Naive_out/naive.parts.out",
            shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        c = "python rendering/render.py"
        subprocess.check_call(
            c + " MPI/MPI_out/mpi.parts.out rendering/mpi_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def run_cuda_opt(number_of_particles: int,
                 particle_initialization_seed: int,
                 cutoff: float,
                 correctness_check: bool = False,
                 rendering: bool = False) -> float:

    s = subprocess.check_output("CUDA/CUDA_out/OPT -n {0} -s {1} -o CUDA/CUDA_out/opt.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    output = s.decode("utf-8")
    print("OPT - " + output)

    if correctness_check is True:
        c = "python correctness-check/correctness-check.py"
        subprocess.check_call(
            c + " CUDA/CUDA_out/opt.parts.out Naive/Naive_out/naive.parts.out",
            shell=True)
        print("TRUE <-- Correctness-check")

    if rendering is True:
        c = "python rendering/render.py"
        subprocess.check_call(
            c + " CUDA/CUDA_out/opt.parts.out rendering/opt_particles_{0}.gif {1}"
            .format(number_of_particles, cutoff), shell=True)

    return float(output.split(" ")[3])


def make_directory(dir_name: str):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":

    number_of_simulations: int = 2  # NUMBER_OF_SIMULATIONS
    number_of_particles_per_simulation: List[int] = [2**j for j in range(10, 10 + number_of_simulations)]  # [4096, ...]
    particle_initialization_seed: int = 42  # PARTICLE_INITIALIZATION_SEED
    cutoff: float = 0.01  # CUTOFF
    max_number_of_processes: int = 6  # MAX_NUMBER_OF_PROCESSES
    correctness_check: bool = True  # CORRECTNESS_CHECK
    rendering: bool = True  # RENDERING
    heap_profiler: bool = False  # HEAP_PROFILER

    for name in ["results", "results/mpi", "results/openmp",
                 "heap-profiler", "heap-profiler/mpi", "heap-profiler/openmp", "heap-profiler/serial"]:
        make_directory(name)

    naive_times: List[float] = []
    serial_times: List[float] = []
    openmp_times: List[List[float]] = []
    mpi_times: List[List[float]] = []
    cuda_opt_times: List[float] = []

    for number_of_particles in number_of_particles_per_simulation:
        if correctness_check is True:
            naive_times.append(
                run_naive(number_of_particles, particle_initialization_seed, cutoff,
                          rendering))

        serial_times.append(
            run_serial(number_of_particles, particle_initialization_seed, cutoff,
                       correctness_check, rendering, heap_profiler))

        openmp_times.append(
            [run_openmp(i, number_of_particles, particle_initialization_seed, cutoff,
                        correctness_check, rendering, heap_profiler)
             for i in range(1, max_number_of_processes * 2 + 1)])

        mpi_times.append(
            [run_mpi(j, number_of_particles, particle_initialization_seed, cutoff,
                     correctness_check, rendering, heap_profiler)
             for j in range(1, max_number_of_processes + 1)])

        cuda_opt_times.append(
            run_cuda_opt(number_of_particles, particle_initialization_seed, cutoff,
                         correctness_check, rendering))

    np.savetxt("results/naive_times.csv", X=np.array(naive_times),
               header=str(number_of_particles_per_simulation), delimiter=",")
    np.savetxt("results/openmp/openmp_times.csv", X=np.array(openmp_times),
               header=str(number_of_particles_per_simulation), delimiter=",")
    np.savetxt("results/mpi/mpi_times.csv", X=np.array(mpi_times),
               header=str(number_of_particles_per_simulation), delimiter=",")
    timing_results = np.block([np.array(serial_times)[:, np.newaxis],
                               np.array([min(times) for times in openmp_times])[:, np.newaxis],
                               np.array([min(times) for times in mpi_times])[:, np.newaxis],
                               np.array(cuda_opt_times)[:, np.newaxis]])
    np.savetxt("results/timing_results.csv", X=timing_results,
               header="serial, openmp, mpi, cuda_opt", delimiter=",")
