#include "common.h"
#include <cuda.h>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;
double min_bin_size = 1.5 * cutoff;
int x_bin_count;
int bin_count;
double bin_size;
int* cellnos;
int* cell_starts;
int* cell_ends;
thrust::device_ptr<particle_t> device_ptr_parts;
thrust::device_ptr<int> device_ptr_cellnos;
thrust::device_ptr<int> device_ptr_cell_starts;
thrust::device_ptr<int> device_ptr_cell_ends;

// region Given functions
__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

//__global__ void compute_forces_gpu(particle_t* particles, int num_parts) {
//    // Get thread (particle) ID
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    if (tid >= num_parts)
//        return;
//
//    particles[tid].ax = particles[tid].ay = 0;
//    for (int j = 0; j < num_parts; j++)
//        apply_force_gpu(particles[tid], particles[j]);
//}

//__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
//
//    // Get thread (particle) ID
//    int tid = threadIdx.x + blockIdx.x * blockDim.x;
//    if (tid >= num_parts)
//        return;
//
//    particle_t* p = &particles[tid];
//    //
//    //  slightly simplified Velocity Verlet integration
//    //  conserves energy better than explicit Euler method
//    //
//    p->vx += p->ax * dt;
//    p->vy += p->ay * dt;
//    p->x += p->vx * dt;
//    p->y += p->vy * dt;
//
//    //
//    //  bounce from walls
//    //
//    while (p->x < 0 || p->x > size) {
//        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
//        p->vx = -(p->vx);
//    }
//    while (p->y < 0 || p->y > size) {
//        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
//        p->vy = -(p->vy);
//    }
//}
// endregion

// region New functions
#define CLAMP(_v, _l, _h) ( (_v) < (_l) ? (_l) : (((_v) > (_h)) ? (_h) : (_v)) )


__device__ static inline int get_cell_for_particle (particle_t & p, double bin_size, int x_bin_count) {
    int x = (int)(p.x / bin_size);
    int y = (int)(p.y / bin_size);
    x = CLAMP(x, 0, x_bin_count - 1);
    y = CLAMP(y, 0, x_bin_count - 1);
    return x + y * x_bin_count;
}


__global__ void assign_cells (particle_t* parts, int* cellnos, int num_parts, double bin_size, int x_bin_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= num_parts) return;

    particle_t& p = parts[tid];
    cellnos[tid] = get_cell_for_particle(p, bin_size, x_bin_count);
}

__global__ void build_index (int* cellnos, int* cell_starts, int* cell_ends, int num_parts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts) return;
    if (tid == 0) {
        cell_starts[cellnos[tid]] = tid;
        return;
    }
    if (cellnos[tid] != cellnos[tid-1]) {
        cell_starts[cellnos[tid]] = tid;
        cell_ends[cellnos[tid-1]] = tid-1;
    }
}

__global__ void compute_forces_gpu(particle_t* parts, int num_parts, int* cell_starts, int* cell_ends, double bin_size, int x_bin_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= num_parts) return;
    parts[tid].ax = parts[tid].ay = 0;
    // Could we be using a 2D grid and skip this?
    int xy = get_cell_for_particle(parts[tid], bin_size, x_bin_count);
    int x = xy % x_bin_count;
    int y = xy / x_bin_count;
    int yl = CLAMP(y - 1, 0, x_bin_count - 1);
    int yh = CLAMP(y + 1, 0, x_bin_count - 1);
    int xl = CLAMP(x - 1, 0, x_bin_count - 1);
    int xh = CLAMP(x + 1, 0, x_bin_count - 1);

    for (int y = yl; y <= yh; y++) {
        for (int x = xl; x <= xh; x++) {
            int first = cell_starts[x + y * x_bin_count];
            if (first == -1) continue;
            int last = cell_ends[x + y * x_bin_count];
            for (int j = first; j <= last; j++) {
                apply_force_gpu(parts[tid], parts[j]);
            }
        }
    }
}

__global__ void move_gpu (particle_t* parts, int* cellnos, int num_parts, double size, double bin_size, int x_bin_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= num_parts) return;
    particle_t * p = &parts[tid];
    // slightly simplified Velocity Verlet integration
    // conserves energy better than explicit Euler method
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;
    // bounce from walls
    while( p->x < 0 || p->x > size ) {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size ) {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }
    cellnos[tid] = get_cell_for_particle(*p, bin_size, x_bin_count);
}
// endregion

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    x_bin_count = (int)(size / min_bin_size);
    bin_count = x_bin_count * x_bin_count;
    bin_size = size / x_bin_count;
    cudaMalloc((void**)&cellnos, num_parts * sizeof(int));
    cudaMalloc((void**)&cell_starts, bin_count * sizeof(int));
    cudaMalloc((void**)&cell_ends, bin_count * sizeof(int));
    assign_cells<<<blks, NUM_THREADS>>>(parts, cellnos, num_parts, bin_size, x_bin_count);
//    int* h_cellnos = new int[num_parts];
//    cudaMemcpy(h_cellnos, cellnos, num_parts * sizeof(int), cudaMemcpyDeviceToHost);
//    for(int i = 0; i < num_parts; ++i)
//        printf("i: %d cellnos[i]: %d\n", i, h_cellnos[i]);
    device_ptr_cellnos = thrust::device_pointer_cast(cellnos);
    device_ptr_cell_starts = thrust::device_pointer_cast(cell_starts);
    device_ptr_cell_ends = thrust::device_pointer_cast(cell_ends);
    device_ptr_parts = thrust::device_pointer_cast(parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // region Rewrite this function
    thrust::sort_by_key(thrust::device,
                        device_ptr_cellnos,
                        device_ptr_cellnos + num_parts,
                        device_ptr_parts);
    thrust::fill(thrust::device,
                 device_ptr_cell_starts,
                 device_ptr_cell_starts + bin_count,
                 -1);
    thrust::fill(thrust::device,
                 device_ptr_cell_ends,
                 device_ptr_cell_ends + bin_count,
                 num_parts - 1);
    build_index<<<blks, NUM_THREADS>>>(cellnos, cell_starts, cell_ends, num_parts);
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, cell_starts, cell_ends, bin_size, x_bin_count);
    move_gpu<<<blks, NUM_THREADS>>>(parts, cellnos, num_parts, size, bin_size, x_bin_count);
    // endregion

//    // Compute forces
//    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts);
//
//    // Move particles
//    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
}

