import subprocess
import os


def run_naive():
    # store output of the program as a byte string in s
    s = subprocess.check_output("Naive/Naive_out/Naive -n 2000 -s 42 -o Naive/Naive_out/naive.parts.out",
                                shell=True)
    # decode s to a normal string
    print("Naive - " + s.decode("utf-8"))


def run_serial():
    s = subprocess.check_output("Serial/Serial_out/Serial -n 2000 -s 42 -o Serial/Serial_out/serial.parts.out",
                                shell=True)
    c = subprocess.check_call("python correctness-check/correctness-check.py Serial/Serial_out/serial.parts.out Naive/Naive_out/naive.parts.out",
                              shell=True)
    print("Serial - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


def run_openmp():
    s = subprocess.check_output("OpenMP/OpenMP_out/OpenMP -n 2000 -s 42 -o OpenMP/OpenMP_out/openmp.parts.out",
                                shell=True)
    c = subprocess.check_call("python correctness-check/correctness-check.py OpenMP/OpenMP_out/openmp.parts.out Naive/Naive_out/naive.parts.out",
                              shell=True)
    print("OpenMP - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


def run_mpi():
    s = subprocess.check_output("mpirun -n 6 MPI/MPI_out/MPI -n 2000 -s 42 -o MPI/MPI_out/mpi.parts.out",
                                shell=True)
    c = subprocess.check_call("python correctness-check/correctness-check.py MPI/MPI_out/mpi.parts.out Naive/Naive_out/naive.parts.out",
                              shell=True)
    print("MPI - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


def run_cuda():
    s = subprocess.check_output("CUDA/CUDA_out/CUDA -n 2000 -s 42 -o CUDA/CUDA_out/CUDA/cuda.parts.out",
                                shell=True)
    c = subprocess.check_call("python correctness-check/correctness-check.py CUDA/CUDA_out/CUDA/cuda.parts.out Naive/Naive_out/naive.parts.out",
                              shell=True)
    print("CUDA - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


def run_opt():
    s = subprocess.check_output("CUDA/CUDA_out/OPT -n 2000 -s 42 -o CUDA/CUDA_out/OPT/opt.parts.out",
                                shell=True)
    c = subprocess.check_call("python correctness-check/correctness-check.py CUDA/CUDA_out/OPT/opt.parts.out Naive/Naive_out/naive.parts.out",
                              shell=True)
    print("OPT - " + s.decode("utf-8") + "    Correctness-check --> TRUE")


if __name__ == "__main__":
    run_naive()
    run_serial()
    run_openmp()
    #run_mpi()
    #run_cuda()
    run_opt()

