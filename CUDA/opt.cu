#include "common.h"
#include <cuda.h>
#include <stdio.h>
#define NUM_THREADS 256

// Put any static global variables here that you will use throughout the simulation.
int blks;

int bins_per_row;
int bin_count;
double bin_size;

int* heads;
int* linked_list;

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

/*
       0   1
     +---X---->
   + +---+---+
0  | | 0 | 1 |
   Y +-------+
1  | | 2 | 3 |
   | +---+---+
   v
*/

__device__ int inline get_bin_id(particle_t& particle, int bins_per_row, int bin_count, double bin_size) {
    int x, y;
    y = particle.y / bin_size;
    x = particle.x / bin_size;
    if (x == bins_per_row) {
        x--;
    }
    if (y == bins_per_row) {
        y--;
    }
    return y * bins_per_row + x;
}


__global__ void rebin(particle_t* particles, int num_parts, int bins_per_row, int bin_count, double bin_size, int* heads, int* linked_list) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < bin_count) {
        heads[tid] = -1;
    }
    if (tid >= num_parts)
        return;
    int bin_id = get_bin_id(particles[tid], bins_per_row, bin_count, bin_size);
    linked_list[tid] = atomicExch(&heads[bin_id], tid);
}

__device__ bool inline has_up(int bin_id, int bins_per_row, int bin_count, double bin_size) {
    return bin_id - bins_per_row > -1;
}
__device__ bool inline has_down(int bin_id, int bins_per_row, int bin_count, double bin_size) {
    return bin_id + bins_per_row < bin_count;
}
__device__ bool inline has_left(int bin_id, int bins_per_row, int bin_count, double bin_size) {
    return bin_id % bins_per_row != 0;
}
__device__ bool inline has_right(int bin_id, int bins_per_row, int bin_count, double bin_size) {
    return bin_id % bins_per_row != bins_per_row - 1;
}

__device__ void inline loop(particle_t* parts, int i, int another_bin_id, int* heads, int* linked_list) {
    int ptr = heads[another_bin_id];
    for (; ptr != -1; ptr = linked_list[ptr]) {
        apply_force_gpu(parts[i], parts[ptr]);
    }
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int bins_per_row, int bin_count, double bin_size, int* heads, int* linked_list) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;
    int bin_id = get_bin_id(particles[tid], bins_per_row, bin_count, bin_size);
    // self
    loop(particles, tid, bin_id, heads, linked_list);
    // up
    if (has_up(bin_id, bins_per_row, bin_count, bin_size)) {
        loop(particles, tid, bin_id - bins_per_row, heads, linked_list);
    }
    // up right
    if (has_up(bin_id, bins_per_row, bin_count, bin_size) && has_right(bin_id, bins_per_row, bin_count, bin_size)) {
        loop(particles, tid, bin_id - bins_per_row + 1, heads, linked_list);
    }
    // right
    if (has_right(bin_id, bins_per_row, bin_count, bin_size)) {
        loop(particles, tid, bin_id + 1, heads, linked_list);
    }
    // down right
    if (has_down(bin_id, bins_per_row, bin_count, bin_size) && has_right(bin_id, bins_per_row, bin_count, bin_size)) {
        loop(particles, tid, bin_id + bins_per_row + 1, heads, linked_list);
    }
    // down
    if (has_down(bin_id, bins_per_row, bin_count, bin_size)) {
        loop(particles, tid, bin_id + bins_per_row, heads, linked_list);
    }
    // down left
    if (has_down(bin_id, bins_per_row, bin_count, bin_size) && has_left(bin_id, bins_per_row, bin_count, bin_size)) {
        loop(particles, tid, bin_id + bins_per_row - 1, heads, linked_list);
    }
    // left
    if (has_left(bin_id, bins_per_row, bin_count, bin_size)) {
        loop(particles, tid, bin_id - 1, heads, linked_list);
    }
    // up left
    if (has_up(bin_id, bins_per_row, bin_count, bin_size) && has_left(bin_id, bins_per_row, bin_count, bin_size)) {
        loop(particles, tid, bin_id - bins_per_row - 1, heads, linked_list);
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    bins_per_row = size / cutoff;
    bin_count = bins_per_row * bins_per_row;
    bin_size = size / bins_per_row;
    cudaMalloc((void**)&heads, bin_count * sizeof(int));
    cudaMalloc((void**)&linked_list, num_parts * sizeof(int));
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // parts live in GPU memory
    // Rewrite this function
    int tmp_blks = (bin_count + NUM_THREADS - 1) / NUM_THREADS;
    rebin<<<tmp_blks, NUM_THREADS>>>(parts, num_parts, bins_per_row, bin_count, bin_size, heads, linked_list);

    // Compute forces
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, bins_per_row, bin_count, bin_size, heads, linked_list);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);

}