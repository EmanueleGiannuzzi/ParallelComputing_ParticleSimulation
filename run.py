import subprocess


def run_naive(number_of_particles: int, particle_initialization_seed: int, cutoff: float) -> str:
    # store output of the program as a byte string in s
    s = subprocess.check_output("Naive/Naive_out/Naive -n {0} -s {1} -o Naive/Naive_out/naive.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    r = subprocess.check_call(
        "python rendering/render.py Naive/Naive_out/naive.parts.out rendering/naive_particles.gif {0}"
        .format(cutoff), shell=True)
    # decode s to a normal string
    output = s.decode("utf-8")
    print("Naive - " + output)
    return output.split(" ")[3]


def run_serial(number_of_particles: int, particle_initialization_seed: int, cutoff: float) -> str:
    s = subprocess.check_output("Serial/Serial_out/Serial -n {0} -s {1} -o Serial/Serial_out/serial.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py Serial/Serial_out/serial.parts.out Naive/Naive_out/naive.parts.out",
                              shell=True)
    r = subprocess.check_call(
        "python rendering/render.py Serial/Serial_out/serial.parts.out rendering/serial_particles.gif {0}"
        .format(cutoff), shell=True)
    output = s.decode("utf-8")
    print("Serial - " + output + "    TRUE <-- Correctness-check")
    return output.split(" ")[3]


def run_openmp(number_of_particles: int, particle_initialization_seed: int, cutoff: float) -> str:
    s = subprocess.check_output("OpenMP/OpenMP_out/OpenMP -n {0} -s {1} -o OpenMP/OpenMP_out/openmp.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py OpenMP/OpenMP_out/openmp.parts.out Naive/Naive_out/naive.parts.out",
        shell=True)
    r = subprocess.check_call(
        "python rendering/render.py OpenMP/OpenMP_out/openmp.parts.out rendering/openmp_particles.gif {0}"
        .format(cutoff), shell=True)
    output = s.decode("utf-8")
    print("OpenMP - " + output + "    TRUE <-- Correctness-check")
    return output.split(" ")[3]


def run_mpi(number_of_processes: int, number_of_particles: int, particle_initialization_seed: int, cutoff: float) -> str:
    s = subprocess.check_output("mpirun -n {0} MPI/MPI_out/MPI -n {1} -s {2} -o MPI/MPI_out/mpi.parts.out"
                                .format(number_of_processes, number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py MPI/MPI_out/mpi.parts.out Naive/Naive_out/naive.parts.out",
        shell=True)
    r = subprocess.check_call(
        "python rendering/render.py MPI/MPI_out/mpi.parts.out rendering/mpi_particles.gif {0}"
        .format(cutoff), shell=True)
    output = s.decode("utf-8")
    print("MPI - " + output + "    TRUE <-- Correctness-check")
    return output.split(" ")[3]


def run_cuda(number_of_particles: int, particle_initialization_seed: int, cutoff: float) -> str:
    s = subprocess.check_output("CUDA/CUDA_out/CUDA -n {0} -s {1} -o CUDA/CUDA_out/cuda.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py CUDA/CUDA_out/cuda.parts.out Naive/Naive_out/naive.parts.out",
        shell=True)
    r = subprocess.check_call(
        "python rendering/render.py CUDA/CUDA_out/cuda.parts.out rendering/cuda_particles.gif {0}"
        .format(cutoff), shell=True)
    output = s.decode("utf-8")
    print("CUDA - " + output + "    TRUE <-- Correctness-check")
    return output.split(" ")[3]


def run_cuda_opt(number_of_particles: int, particle_initialization_seed: int, cutoff: float) -> str:
    s = subprocess.check_output("CUDA/CUDA_out/OPT -n {0} -s {1} -o CUDA/CUDA_out/opt.parts.out"
                                .format(number_of_particles, particle_initialization_seed),
                                shell=True)
    c = subprocess.check_call(
        "python correctness-check/correctness-check.py CUDA/CUDA_out/opt.parts.out Naive/Naive_out/naive.parts.out",
        shell=True)
    r = subprocess.check_call(
        "python rendering/render.py CUDA/CUDA_out/opt.parts.out rendering/opt_particles.gif {0}"
        .format(cutoff), shell=True)
    output = s.decode("utf-8")
    print("OPT - " + output + "    TRUE <-- Correctness-check")
    return output.split(" ")[3]


if __name__ == "__main__":
    number_of_particles: int = 1000
    particle_initialization_seed: int = 42
    cutoff: float = 0.01
    number_of_processes: int = 6

    naive_time = run_naive(number_of_particles, particle_initialization_seed, cutoff)
    serial_time = run_serial(number_of_particles, particle_initialization_seed, cutoff)
    openmp_time = run_openmp(number_of_particles, particle_initialization_seed, cutoff)
    mpi_time = [run_mpi(i, number_of_particles, particle_initialization_seed, cutoff)
                for i in range(1, number_of_processes + 1)]
    cuda_time = run_cuda(number_of_particles, particle_initialization_seed, cutoff)
    cuda_opt_time = run_cuda_opt(number_of_particles, particle_initialization_seed, cutoff)

