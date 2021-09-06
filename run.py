import subprocess
import os

number_of_particles = 4000
particle_initialization_seed = 42
cutoff = 0.01

def run_naive():
    # store output of the program as a byte string in s
    s = subprocess.check_output("Naive/Naive_out/Naive -n {0} -s {1} -o Naive/Naive_out/naive.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    r = subprocess.check_call(
        "python rendering/render.py Naive/Naive_out/naive.parts.out rendering/naive_particles.gif {0}"
        .format(cutoff), shell=True)
    # decode s to a normal string
    print("Naive - " + s.decode("utf-8"))


def run_serial():
    s = subprocess.check_output("Serial/Serial_out/Serial -n {0} -s {1} -o Serial/Serial_out/serial.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py Serial/Serial_out/serial.parts.out Naive/Naive_out/naive.parts.out",
                              shell=True)
    r = subprocess.check_call(
        "python rendering/render.py Serial/Serial_out/serial.parts.out rendering/serial_particles.gif {0}"
        .format(cutoff), shell=True)
    print("Serial - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


def run_openmp():
    s = subprocess.check_output("OpenMP/OpenMP_out/OpenMP -n {0} -s {1} -o OpenMP/OpenMP_out/openmp.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py OpenMP/OpenMP_out/openmp.parts.out Naive/Naive_out/naive.parts.out",
        shell=True)
    r = subprocess.check_call(
        "python rendering/render.py OpenMP/OpenMP_out/openmp.parts.out rendering/openmp_particles.gif {0}"
        .format(cutoff), shell=True)
    print("OpenMP - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


def run_mpi():
    s = subprocess.check_output("mpirun -n 2 MPI/MPI_out/MPI -n {0} -s {1} -o MPI/MPI_out/mpi.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py MPI/MPI_out/mpi.parts.out Naive/Naive_out/naive.parts.out",
        shell=True)
    r = subprocess.check_call(
        "python rendering/render.py MPI/MPI_out/mpi.parts.out rendering/mpi_particles.gif {0}"
        .format(cutoff), shell=True)
    print("MPI - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


def run_cuda():
    s = subprocess.check_output("CUDA/CUDA_out/CUDA -n {0} -s {1} -o CUDA/CUDA_out/cuda.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py CUDA/CUDA_out/cuda.parts.out Naive/Naive_out/naive.parts.out",
        shell=True)
    r = subprocess.check_call(
        "python rendering/render.py CUDA/CUDA_out/cuda.parts.out rendering/cuda_particles.gif {0}"
        .format(cutoff), shell=True)
    print("CUDA - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


def run_opt():
    s = subprocess.check_output("CUDA/CUDA_out/OPT -n {0} -s {1} -o CUDA/CUDA_out/opt.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py CUDA/CUDA_out/opt.parts.out Naive/Naive_out/naive.parts.out",
        shell=True)
    r = subprocess.check_call(
        "python rendering/render.py CUDA/CUDA_out/opt.parts.out rendering/opt_particles.gif {0}"
        .format(cutoff), shell=True)
    print("OPT - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


if __name__ == "__main__":
    run_naive()
    run_serial()
    run_openmp()
    #run_mpi()
    run_cuda()
    run_opt()

