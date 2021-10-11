import subprocess
from typing import List
import numpy as np
import os

render: str = "python rendering/render.py "
correct: str = "python correctness-check/correctness-check.py "
massif: str = "valgrind --tool=massif --massif-out-file=heap-profiler/"
houston: str = "Houston, We’ve Had a Problem Here!"
correct_parts_out: str = " Naive/Naive_out/naive.parts.out.{0}"  # " Serial/Serial_out/serial.parts.out.{0}"


def run_naive(_number_of_particles: int,
              _particle_initialization_seed: int,
              _cutoff: float,
              _rendering: bool = False) -> float:

    naive_parts_out: str = "Naive/Naive_out/naive.parts.out.{0}".format(_number_of_particles)
    timing: float
    if not os.path.isfile(naive_parts_out) or not os.path.isfile(naive_parts_out + ".timing.npy"):
        s = subprocess.check_output("Naive/Naive_out/Naive -n {0} -s {1} -o Naive/Naive_out/naive.parts.out.{0}"
                                    .format(_number_of_particles, _particle_initialization_seed), shell=True)
        output = s.decode("utf-8")
        print("Naive - " + output)
        timing = float(output.split(" ")[3])
        np.save(naive_parts_out + ".timing", timing)
    else:
        print(naive_parts_out + " already exists! -> Skipping simulation")
        timing = np.load(naive_parts_out + ".timing.npy")

    if _rendering is True:
        subprocess.check_call(
            render + naive_parts_out + " rendering/naive_particles_{0}.gif {1}"
            .format(_number_of_particles, _cutoff), shell=True)

    return timing


def run_serial(_number_of_particles: int,
               _particle_initialization_seed: int,
               _cutoff: float,
               _correctness_check: bool = False,
               _rendering: bool = False,
               _heap_profiler: bool = False) -> float:

    serial_parts_out: str = "Serial/Serial_out/serial.parts.out.{0}".format(_number_of_particles)
    timing: float
    if not os.path.isfile(serial_parts_out) or not os.path.isfile(serial_parts_out + ".timing.npy"):
        c = "Serial/Serial_out/Serial -n {0} -s {1} -o " + serial_parts_out
        s = subprocess.check_output(
            (c if not _heap_profiler else massif + "serial/massif.out.{0} " + c).format(
                _number_of_particles, _particle_initialization_seed), shell=True)
        output = s.decode("utf-8")
        print("Serial - " + output)
        timing = float(output.split(" ")[3])
        np.save(serial_parts_out + ".timing", timing)
    else:
        print(serial_parts_out + " already exists! -> Skipping simulation")
        timing = np.load(serial_parts_out + ".timing.npy")

    if _rendering is True:
        subprocess.check_call(
            render + " " + serial_parts_out + " rendering/serial_particles_{0}.gif {1}"
            .format(_number_of_particles, _cutoff), shell=True)

    if _correctness_check is True:
        try:
            subprocess.check_call(
                correct + serial_parts_out + correct_parts_out.format(_number_of_particles), shell=True)
        except subprocess.CalledProcessError:
            print(houston)

    return timing


def run_openmp(_number_of_threads: int,
               _number_of_particles: int,
               _particle_initialization_seed: int,
               _cutoff: float,
               _correctness_check: bool = False,
               _rendering: bool = False,
               _heap_profiler: bool = False) -> float:

    openmp_parts_out: str = "OpenMP/OpenMP_out/openmp.parts.out.{0}_{1}".format(_number_of_particles, _number_of_threads)
    timing: float
    if not os.path.isfile(openmp_parts_out) or not os.path.isfile(openmp_parts_out + ".timing.npy"):
        my_env = os.environ.copy()
        my_env['OMP_NUM_THREADS'] = str(_number_of_threads)
        c = "OpenMP/OpenMP_out/OpenMP -n {0} -s {1} -o " + openmp_parts_out
        s = subprocess.check_output(
            (c if not _heap_profiler else massif + "openmp/massif.out.{0}_{2} " + c).format(
                _number_of_particles, _particle_initialization_seed, _number_of_threads), shell=True, env=my_env)
        output = s.decode("utf-8")
        print("OpenMP with {0} threads - ".format(_number_of_threads) + output)
        timing = float(output.split(" ")[3])
        np.save(openmp_parts_out + ".timing", timing)
    else:
        print(openmp_parts_out + " already exists! -> Skipping simulation")
        timing = np.load(openmp_parts_out + ".timing.npy")

    if _rendering is True:
        subprocess.check_call(
            render + " " + openmp_parts_out + " rendering/openmp_particles_{0}.gif {1}"
            .format(_number_of_particles, _cutoff), shell=True)

    if _correctness_check is True:
        try:
            subprocess.check_call(
                correct + openmp_parts_out + correct_parts_out.format(_number_of_particles), shell=True)
        except subprocess.CalledProcessError:
            print(houston)

    return timing


def run_mpi(number_of_processes: int,
            _number_of_particles: int,
            _particle_initialization_seed: int,
            _cutoff: float,
            _correctness_check: bool = False,
            _rendering: bool = False,
            _heap_profiler: bool = False) -> float:

    mpi_parts_out: str = "MPI/MPI_out/mpi.parts.out.{0}_{1}".format(_number_of_particles, number_of_processes)
    timing: float
    if not os.path.isfile(mpi_parts_out) or not os.path.isfile(mpi_parts_out + ".timing.npy"):
        c1 = "mpirun -n {0}"
        c2 = " MPI/MPI_out/MPI -n {1} -s {2} -o " + mpi_parts_out
        s = subprocess.check_output(
            (c1 + c2 if not _heap_profiler else
                c1 + " " + massif + "mpi/massif.out.{1}_{0}" + c2).format(
                number_of_processes, _number_of_particles, _particle_initialization_seed), shell=True)
        output = s.decode("utf-8")
        print("MPI with {0} processes - ".format(number_of_processes) + output)
        timing = float(output.split(" ")[3])
        np.save(mpi_parts_out + ".timing", timing)
    else:
        print(mpi_parts_out + " already exists! -> Skipping simulation")
        timing = np.load(mpi_parts_out + ".timing.npy")

    if _rendering is True:
        subprocess.check_call(
            render + " " + mpi_parts_out + " rendering/mpi_particles_{0}.gif {1}"
            .format(_number_of_particles, _cutoff), shell=True)

    if _correctness_check is True:
        try:
            subprocess.check_call(
                correct + mpi_parts_out + correct_parts_out.format(_number_of_particles), shell=True)
        except subprocess.CalledProcessError:
            print(houston)

    return timing


def run_cuda(_number_of_particles: int,
             _particle_initialization_seed: int,
             _cutoff: float,
             _correctness_check: bool = False,
             _rendering: bool = False) -> float:

    cuda_parts_out: str = "CUDA/CUDA_out/cuda.parts.out.{0}".format(_number_of_particles)
    timing: float
    if not os.path.isfile(cuda_parts_out) or not os.path.isfile(cuda_parts_out + ".timing.npy"):
        s = subprocess.check_output(("CUDA/CUDA_out/CUDA -n {0} -s {1} -o " + cuda_parts_out).format(
            _number_of_particles, _particle_initialization_seed), shell=True)
        output = s.decode("utf-8")
        print("CUDA - " + output)
        timing = float(output.split(" ")[3])
        np.save(cuda_parts_out + ".timing", timing)
    else:
        print(cuda_parts_out + " already exists! -> Skipping simulation")
        timing = np.load(cuda_parts_out + ".timing.npy")

    if _rendering is True:
        subprocess.check_call(
            render + " " + cuda_parts_out + " rendering/cuda_particles_{0}.gif {1}"
            .format(_number_of_particles, _cutoff), shell=True)

    if _correctness_check is True:
        try:
            subprocess.check_call(
                correct + cuda_parts_out + correct_parts_out.format(_number_of_particles), shell=True)
        except subprocess.CalledProcessError:
            print(houston)

    return timing


def run_cuda_opt(_number_of_particles: int,
                 _particle_initialization_seed: int,
                 _cutoff: float,
                 _correctness_check: bool = False,
                 _rendering: bool = False) -> float:

    cuda_opt_parts_out: str = "CUDA/CUDA_out/opt.parts.out.{0}".format(_number_of_particles)
    timing: float
    if not os.path.isfile(cuda_opt_parts_out) or not os.path.isfile(cuda_opt_parts_out + ".timing.npy"):
        s = subprocess.check_output(("CUDA/CUDA_out/OPT -n {0} -s {1} -o " + cuda_opt_parts_out).format(
            _number_of_particles, _particle_initialization_seed), shell=True)
        output = s.decode("utf-8")
        print("OPT - " + output)
        timing = float(output.split(" ")[3])
        np.save(cuda_opt_parts_out + ".timing", timing)
    else:
        print(cuda_opt_parts_out + " already exists! -> Skipping simulation")
        timing = np.load(cuda_opt_parts_out + ".timing.npy")

    if _rendering is True:
        subprocess.check_call(
            render + " " + cuda_opt_parts_out + " rendering/opt_particles_{0}.gif {1}"
            .format(_number_of_particles, _cutoff), shell=True)

    if _correctness_check is True:
        try:
            subprocess.check_call(
                correct + cuda_opt_parts_out + correct_parts_out.format(_number_of_particles), shell=True)
        except subprocess.CalledProcessError:
            print(houston)

    return timing


def make_directory(dir_name: str):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":

    number_of_simulations: int = 6  # NUMBER_OF_SIMULATIONS
    initial_power_of_2: int = 12  # INITIAL_POWER_OF_2
    np.save("results/initial_power_of_2", initial_power_of_2)
    number_of_particles_per_simulation: List[int] = [2**j for j in range(initial_power_of_2,
                                                                         initial_power_of_2 + number_of_simulations)]
    particle_initialization_seed: int = 42  # PARTICLE_INITIALIZATION_SEED
    cutoff: float = 0.01  # CUTOFF
    max_number_of_processes: int = 6  # MAX_NUMBER_OF_PROCESSES
    max_number_of_threads: int = max_number_of_processes * 2  # MAX_NUMBER_OF_THREADS
    correctness_check: bool = False  # CORRECTNESS_CHECK
    rendering: bool = False  # RENDERING
    heap_profiler: bool = False  # HEAP_PROFILER

    for name in ["results", "results/mpi", "results/openmp",
                 "heap-profiler", "heap-profiler/mpi", "heap-profiler/openmp", "heap-profiler/serial"]:
        make_directory(name)

    naive_times: List[float] = []
    serial_times: List[float] = []
    openmp_times: List[List[float]] = []
    mpi_times: List[List[float]] = []
    cuda_times: List[float] = []
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
             for i in range(1, max_number_of_threads + 1)])

        mpi_times.append(
            [run_mpi(j, number_of_particles, particle_initialization_seed, cutoff,
                     correctness_check, rendering, heap_profiler)
             for j in range(1, max_number_of_processes + 1)])

        cuda_times.append(
            run_cuda(number_of_particles, particle_initialization_seed, cutoff,
                     correctness_check, rendering))

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
                               np.array(cuda_times)[:, np.newaxis],
                               np.array(cuda_opt_times)[:, np.newaxis]])
    np.savetxt("results/timing_results.csv", X=timing_results,
               header="serial, openmp, mpi, cuda, cuda_opt", delimiter=",")
