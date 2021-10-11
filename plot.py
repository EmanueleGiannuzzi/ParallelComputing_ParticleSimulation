import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

timing_results = np.loadtxt("results/timing_results.csv", delimiter=",")
serial = timing_results[:, 0]
openmp = timing_results[:, 1]
mpi = timing_results[:, 2]
cuda = timing_results[:, 4]

# naive_timings = np.loadtxt("results/naive_times.csv", delimiter=",")

openmp_timings = np.loadtxt("results/openmp/openmp_times.csv", delimiter=",")
max_number_of_threads = openmp_timings.shape[-1]
number_of_threads = [p for p in range(1, max_number_of_threads + 1)]

mpi_timings = np.loadtxt("results/mpi/mpi_times.csv", delimiter=",")
max_number_of_processes = mpi_timings.shape[-1]
number_of_processes = [p for p in range(1, max_number_of_processes + 1)]

number_of_simulations = timing_results.shape[0]
initial_power_of_2 = np.load("results/initial_power_of_2.npy")
number_of_particles_per_simulation = [2**j for j in range(initial_power_of_2,
                                                          initial_power_of_2 + number_of_simulations)]

# region Timing

# plt.close("all")
# # Serial timings
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(1, 1, 1)
# title = ax.set_title("Naive vs Binned Timings", fontsize="x-large")
# ax.set_ylabel('Wallclock time (s)', fontsize="x-large")
# ax.set_xlabel('Number of particles (N)', fontsize="x-large")
# serial_timings = [naive_timings, serial]
# for timing, paradigm in zip(serial_timings, ["Naive", "Binned"]):
#     ax.plot(number_of_particles_per_simulation, timing, label=paradigm)
# ax.legend()
# ax.set_yscale("log")
# ax.yaxis.set_major_formatter(ScalarFormatter())
# plt.xticks(number_of_particles_per_simulation, rotation=45)
# plt.tight_layout()
# fig.canvas.draw()
# plt.savefig("results/serial_timings.png")

plt.close("all")
# Overall timings
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("Overall Timings", fontsize="x-large")
ax.set_ylabel('Wallclock time (s)', fontsize="x-large")
ax.set_xlabel('Number of particles (N)', fontsize="x-large")
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
title = ax.set_title("OpenMP Timings", fontsize="x-large")
ax.set_ylabel("Wallclock time (s)", fontsize="x-large")
ax.set_xlabel("Number of particles (N)", fontsize="x-large")
for timing, threads in zip(np.transpose(openmp_timings), number_of_threads):
    ax.plot(number_of_particles_per_simulation, timing, label=threads)
ax.legend(title="Number of threads (P)", loc=2)
# ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_particles_per_simulation, rotation=45)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/openmp/openmp_timings.png")

plt.close("all")
# MPI timings
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("MPI Timings", fontsize="x-large")
ax.set_ylabel('Wallclock time (s)', fontsize="x-large")
ax.set_xlabel('Number of particles (N)', fontsize="x-large")
for timing, processes in zip(np.transpose(mpi_timings), number_of_processes):
    ax.plot(number_of_particles_per_simulation, timing, label=processes)
ax.legend(title="Number of processes (P)", loc=2)
# ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_particles_per_simulation, rotation=45)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/mpi/mpi_timings.png")

plt.close("all")
# CUDA timings
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("CUDA Timings", fontsize="x-large")
ax.set_ylabel('Wallclock time (s)', fontsize="x-large")
ax.set_xlabel('Number of particles (N)', fontsize="x-large")
ax.plot(number_of_particles_per_simulation, cuda, color="red", linestyle="dashed")
ax.scatter(number_of_particles_per_simulation, cuda, color="red")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_particles_per_simulation, rotation=45)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/cuda/cuda_timings.png")

plt.close("all")
# CUDA vs Serial timings
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("CUDA vs Serial Timings", fontsize="x-large")
ax.set_ylabel('Wallclock time (s)', fontsize="x-large")
ax.set_xlabel('Number of particles (N)', fontsize="x-large")
ax.plot(number_of_particles_per_simulation, cuda, color="red", linestyle="dashed")
ax.scatter(number_of_particles_per_simulation, cuda, color="red")
ax.plot(number_of_particles_per_simulation, serial, color="blue", linestyle="dashed")
ax.scatter(number_of_particles_per_simulation, serial, color="blue")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_particles_per_simulation, rotation=45)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/cuda/cuda_vs_serial_timings.png")
# endregion
# region Overhead

plt.close("all")
# OpenMP overhead
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("OpenMP Overhead", fontsize="x-large")
ax.set_ylabel(r"${t(N,P)}\times{P}\minus{t(N,1)^*}$", fontsize="x-large")
ax.set_xlabel("Number of threads (P)", fontsize="x-large")
openmp_overhead = openmp_timings *\
                  np.repeat(np.expand_dims(number_of_threads, axis=0),
                            repeats=number_of_simulations, axis=0) -\
                  np.reshape(np.repeat(serial, repeats=max_number_of_threads, axis=0), openmp_timings.shape)
for timing, number_of_particles in zip(openmp_overhead, number_of_particles_per_simulation):
    ax.plot(number_of_threads, timing, label=number_of_particles)
ax.legend(title="Number of particles (N)", loc=4)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_threads)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/openmp/openmp_overhead.png")

plt.close("all")
# MPI overhead
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("MPI Overhead", fontsize="x-large")
ax.set_ylabel(r"${t(N,P)}\times{P}\minus{t(N,1)^*}$", fontsize="x-large")
ax.set_xlabel('Number of processes (P)', fontsize="x-large")
mpi_overhead = mpi_timings * \
                  np.repeat(np.expand_dims(number_of_processes, axis=0),
                            repeats=number_of_simulations, axis=0) - \
                  np.reshape(np.repeat(serial, repeats=max_number_of_processes, axis=0), mpi_timings.shape)
for timing, number_of_particles in zip(mpi_overhead, number_of_particles_per_simulation):
    ax.plot(number_of_processes, timing, label=number_of_particles)
ax.legend(title="Number of particles (N)", loc=4)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_processes)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/mpi/mpi_overhead.png")
# endregion
# region Speedup

plt.close("all")
# OpenMP Speedup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("OpenMP Speedup", fontsize="x-large")
ax.set_ylabel(r"${S(N,P)=}\frac{t(N,1)^*}{t(N,P)}$", fontsize="x-large")
ax.set_xlabel('Number of threads (P)', fontsize="x-large")
openmp_speedup = np.repeat(np.expand_dims(serial, axis=1), repeats=12, axis=1) / np.array(openmp_timings)
for speedup, number_of_particles in zip(openmp_speedup, number_of_particles_per_simulation):
    ax.plot(number_of_threads, speedup, label=number_of_particles)
ax.plot(number_of_threads, [1] * max_number_of_threads, linestyle="dashed", color="grey")
ax.legend(title="Number of particles (N)", loc=4)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_threads)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/openmp/openmp_speedup.png")

plt.close("all")
# MPI Speedup
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("MPI Speedup", fontsize="x-large")
ax.set_ylabel(r"${S(N,P)=}\frac{t(N,1)^*}{t(N,P)}$", fontsize="x-large")
ax.set_xlabel('Number of processes (P)', fontsize="x-large")
mpi_speedup = np.repeat(np.expand_dims(serial, axis=1), repeats=6, axis=1) / np.array(mpi_timings)
for speedup, number_of_particles in zip(mpi_speedup, number_of_particles_per_simulation):
    ax.plot(number_of_processes, speedup, label=number_of_particles)
ax.plot(number_of_processes, [1] * max_number_of_processes, linestyle="dashed", color="grey")
ax.legend(title="Number of particles (N)", loc=4)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_processes)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/mpi/mpi_speedup.png")
# endregion
# region Efficiency

plt.close("all")
# OpenMP Efficiency
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("OpenMP Efficiency", fontsize="x-large")
ax.set_ylabel(r"${E(N,P)=}\frac{S(N,P)}{P}$", fontsize="x-large")
ax.set_xlabel('Number of threads (P)', fontsize="x-large")
openmp_efficiency = np.array(openmp_speedup) /\
                    np.repeat(np.expand_dims(number_of_threads, axis=0),
                              repeats=number_of_simulations, axis=0)
for efficiency, number_of_particles in zip(openmp_efficiency, number_of_particles_per_simulation):
    ax.plot(number_of_threads, efficiency, label=number_of_particles)
ax.legend(title="Number of particles (N)")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_threads)
yticks = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
ytickslabels = [str(int(k*100))+"%" for k in yticks]
ax.set_yticks(ticks=yticks)
ax.set_yticklabels(ytickslabels)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/openmp/openmp_efficiency.png")

plt.close("all")
# MPI Efficiency
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("MPI Efficiency", fontsize="x-large")
ax.set_ylabel(r"${E(N,P)=}\frac{S(N,P)}{P}$", fontsize="x-large")
ax.set_xlabel('Number of processes (P)', fontsize="x-large")
mpi_efficiency = np.array(mpi_speedup) /\
                 np.repeat(np.expand_dims(number_of_processes, axis=0),
                           repeats=number_of_simulations, axis=0)
for efficiency, number_of_particles in zip(mpi_efficiency, number_of_particles_per_simulation):
    ax.plot(number_of_processes, efficiency, label=number_of_particles)
ax.legend(title="Number of particles (N)")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_processes)
ax.set_yticks(ticks=yticks)
ax.set_yticklabels(ytickslabels)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/mpi/mpi_efficiency.png")
# endregion
# region Strong Scaling Test

plt.close("all")
# OpenMP Strong Scaling Test
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("OpenMP Strong Scaling Test", fontsize="x-large")
ax.set_ylabel('Wallclock time (s)', fontsize="x-large")
ax.set_xlabel('Number of threads (P)', fontsize="x-large")
for timing, number_of_particles in zip(openmp_timings, number_of_particles_per_simulation):
    ax.plot(number_of_threads, timing, label=number_of_particles)
ax.legend(title="Number of particles (N)", loc=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_threads)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/openmp/openmp_strong_scaling.png")

plt.close("all")
# MPI Strong Scaling Test
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("MPI Strong Scaling Test", fontsize="x-large")
ax.set_ylabel('Wallclock time (s)', fontsize="x-large")
ax.set_xlabel('Number of processes (P)', fontsize="x-large")
for timing, number_of_particles in zip(mpi_timings, number_of_particles_per_simulation):
    ax.plot(number_of_processes, timing, label=number_of_particles)
ax.legend(title="Number of particles (N)", loc=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_processes)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/mpi/mpi_strong_scaling.png")

# OpenMP Strong Scaling Table
openmp_strong_scaling_table = np.hstack([np.array(number_of_threads)[:, np.newaxis],
                                         openmp_timings[-2][:, np.newaxis],
                                         openmp_speedup[-2][:, np.newaxis],
                                         openmp_efficiency[-2][:, np.newaxis]])
np.savetxt("results/openmp_strong_scaling_table.csv", X=np.around(openmp_strong_scaling_table, decimals=2),
           header="P, t(N=16384;P), S(N=16384;P), E(N=16384;P)", delimiter=",")

# MPI Strong Scaling Table
mpi_strong_scaling_table = np.hstack([np.array(number_of_processes)[:, np.newaxis],
                                      mpi_timings[2][:, np.newaxis],
                                      mpi_speedup[2][:, np.newaxis],
                                      mpi_efficiency[2][:, np.newaxis]])
np.savetxt("results/mpi_strong_scaling_table.csv", X=np.around(mpi_strong_scaling_table, decimals=2),
           header="P, t(N=4096;P), S(N=4096;P), E(N=4096;P)", delimiter=",")
# endregion
# region Memory consumption

plt.close("all")
# Heap memory allocation
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
title = ax.set_title("Memory consumption", fontsize="x-large")
ax.set_ylabel("Heap memory size (MiB)", fontsize="x-large")
ax.set_xlabel("Number of particles (N)", fontsize="x-large")
serial_heap_memory_MiB = [0.955, 1.8, 3.5, 6.9, 13.8, 27.5]
openmp_heap_memory_MiB = [1, 2, 3.8, 7.5, 15, 29.9]
mpi_heap_memory_MiB = [4.3, 5.7, 8.5, 14.2, 25.5, 48.4]
for memory_consumption, paradigm in zip([serial_heap_memory_MiB,
                                         openmp_heap_memory_MiB,
                                         mpi_heap_memory_MiB], ["Serial", "OpenMP", "MPI"]):
    ax.plot(number_of_particles_per_simulation, memory_consumption, label=paradigm, linestyle="dashed", alpha=0.75)
    ax.scatter(number_of_particles_per_simulation, memory_consumption, alpha=0.75)
ax.legend()
# ax.set_yscale("log")
# ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(number_of_particles_per_simulation, rotation=45)
plt.tight_layout()
fig.canvas.draw()
plt.savefig("results/heap_memory/serial_openmp_mpi.png")
# endregion
