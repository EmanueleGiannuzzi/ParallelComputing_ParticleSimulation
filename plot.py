import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

timing_results = np.loadtxt("results/timing_results.csv", delimiter=",")
serial = timing_results[:, 0]
openmp = timing_results[:, 1]
mpi = timing_results[:, 2]
cuda = timing_results[:, 3]

# naive_timings = np.loadtxt("results/naive_times.csv", delimiter

openmp_timings = np.loadtxt("results/openmp/openmp_times.csv", delimiter=",")
max_number_of_threads = openmp_timings.shape[-1]
number_of_threads = [p for p in range(1, max_number_of_threads + 1)]

mpi_timings = np.loadtxt("results/mpi/mpi_times.csv", delimiter=",")
max_number_of_processes = mpi_timings.shape[-1]
number_of_processes = [p for p in range(1, max_number_of_processes + 1)]

number_of_simulations = timing_results.shape[0]
number_of_particles_per_simulation = [2**j for j in range(12, 12 + number_of_simulations)]

plt.close("all")
# Overall timings
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("Overall timings", fontsize="x-large")
ax.set_ylabel('Time (s)', fontsize="x-large")
ax.set_xlabel('Number of particles', fontsize="x-large")
overall_timings = [serial, openmp, mpi, cuda]
for timing, paradigm in zip(overall_timings, ["Serial", "OpenMP", "MPI", "CUDA"]):
    ax.plot(number_of_particles_per_simulation, timing, label=paradigm)
ax.legend()
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_particles_per_simulation, rotation=45)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/overall_timings.png")

plt.close("all")
# OpenMP timings
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("OpenMP timings", fontsize="x-large")
ax.set_ylabel('Time (s)', fontsize="x-large")
ax.set_xlabel('Number of threads', fontsize="x-large")
for timing, number_of_particles in zip(openmp_timings, number_of_particles_per_simulation):
    ax.plot(number_of_threads, timing, label=number_of_particles)
ax.legend(loc=1)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/openmp/openmp_timings.png")

plt.close("all")
# MPI timings
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("MPI timings", fontsize="x-large")
ax.set_ylabel('Time (s)', fontsize="x-large")
ax.set_xlabel('Number of processes', fontsize="x-large")
for timing, number_of_particles in zip(mpi_timings, number_of_particles_per_simulation):
    ax.plot(number_of_processes, timing, label=number_of_particles)
ax.legend(loc=1)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/mpi/mpi_timings.png")

plt.close("all")
# OpenMP Speedup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("OpenMP Speedup", fontsize="x-large")
ax.set_ylabel('Speedup (Ts/Tp)', fontsize="x-large")
ax.set_xlabel('Number of threads', fontsize="x-large")
openmp_speedup = np.repeat(np.expand_dims(serial, axis=1), repeats=12, axis=1) / np.array(openmp_timings)
for speedup, number_of_particles in zip(openmp_speedup, number_of_particles_per_simulation):
    ax.plot(number_of_threads, speedup, label=number_of_particles)
ax.legend()
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/openmp/openmp_speedup.png")

plt.close("all")
# MPI Speedup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("MPI Speedup", fontsize="x-large")
ax.set_ylabel('Speedup (Ts/Tp)', fontsize="x-large")
ax.set_xlabel('Number of processes', fontsize="x-large")
mpi_speedup = np.repeat(np.expand_dims(serial, axis=1), repeats=6, axis=1) / np.array(mpi_timings)
for speedup, number_of_particles in zip(mpi_speedup, number_of_particles_per_simulation):
    ax.plot(number_of_processes, speedup, label=number_of_particles)
ax.legend()
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/mpi/mpi_speedup.png")

plt.close("all")
# OpenMP Efficiency
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("OpenMP Efficiency", fontsize="x-large")
ax.set_ylabel('Efficiency (S/p)', fontsize="x-large")
ax.set_xlabel('Number of threads', fontsize="x-large")
openmp_efficiency = np.array(openmp_speedup) / np.repeat(np.expand_dims(number_of_threads, axis=0),
                                                         repeats=number_of_simulations, axis=0)
for efficiency, number_of_particles in zip(openmp_efficiency, number_of_particles_per_simulation):
    ax.plot(number_of_threads, efficiency, label=number_of_particles)
ax.legend()
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/openmp/openmp_efficiency.png")

plt.close("all")
# MPI Efficiency
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("MPI Efficiency", fontsize="x-large")
ax.set_ylabel('Efficiency (S/p)', fontsize="x-large")
ax.set_xlabel('Number of processes', fontsize="x-large")
mpi_efficiency = np.array(mpi_speedup) / np.repeat(np.expand_dims(number_of_processes, axis=0),
                                                   repeats=number_of_simulations, axis=0)
for efficiency, number_of_particles in zip(mpi_efficiency, number_of_particles_per_simulation):
    ax.plot(number_of_processes, efficiency, label=number_of_particles)
ax.legend()
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/mpi/mpi_efficiency.png")


