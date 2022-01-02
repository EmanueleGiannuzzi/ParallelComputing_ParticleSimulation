#include "common.h"
#include <cstdio>
#define NUM_THREADS 256


// Global variables that are used throughout the simulation
unsigned int numBlocks;
unsigned int rebin_numBlocks;

__constant__ int bins_per_row;
__constant__ int bin_count;
__constant__ double bin_size;

int* heads;
int* linked_list;


__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    //r2 = fmax( r2, min_r * min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    // Very simple short-range repulsive force
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

__device__ int inline get_bin_id(particle_t& particle) {
    int x, y;
    y = int(particle.y / bin_size);
    x = int(particle.x / bin_size);
    if (x == bins_per_row)
        x--;
    if (y == bins_per_row)
        y--;
    return y * bins_per_row + x;
}


__global__ void rebin_gpu(particle_t* parts, int num_parts, int* heads_gpu, int* linked_list_gpu) {
    // Get thread (particle) ID
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < bin_count)
        heads_gpu[index] = -1;
    if (index >= num_parts)
        return;
    int bin_id = get_bin_id(parts[index]);
    linked_list_gpu[index] = atomicExch(&heads_gpu[bin_id], index);  // coordination between blocks
}

__device__ bool inline has_up(int bin_id) {
    return bin_id - bins_per_row > -1;
}
__device__ bool inline has_down(int bin_id) {
    return bin_id + bins_per_row < bin_count;
}
__device__ bool inline has_left(int bin_id) {
    return bin_id % bins_per_row != 0;
}
__device__ bool inline has_right(int bin_id) {
    return bin_id % bins_per_row != bins_per_row - 1;
}

__device__ void inline loop(particle_t* parts, int i, int another_bin_id, const int* heads_gpu, const int* linked_list_gpu) {
    int ptr = heads_gpu[another_bin_id];
    for (; ptr != -1; ptr = linked_list_gpu[ptr]) {
        apply_force_gpu(parts[i], parts[ptr]);
    }
}

__global__ void compute_forces_gpu(particle_t* parts, int num_parts, int* heads_gpu, int* linked_list_gpu) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    parts[tid].ax = parts[tid].ay = 0;
    int bin_id = get_bin_id(parts[tid]);

//    int lowi = -1, highi = 1, lowj = -1, highj = 1;
//    if (bin_id < bins_per_row) // in the first row
//        lowj = 0;
//    if (bin_id % bins_per_row == 0) // in the first column
//        lowi = 0;
//    if (bin_id % bins_per_row == (bins_per_row-1))
//        highi = 0;
//    if (bin_id >= bins_per_row*(bins_per_row-1))
//        highj = 0;
//
//    for (int i = lowi; i <= highi; i++){
//        for (int j = lowj; j <= highj; j++){
//            int neighborBin = bin_id + i + bins_per_row*j;
//            loop(parts, tid, neighborBin, heads, linked_list);
//        }
//    }

    // self
    loop(parts, tid, bin_id, heads_gpu, linked_list_gpu);

    // up
    if (has_up(bin_id)) {
        loop(parts, tid, bin_id - bins_per_row, heads_gpu, linked_list_gpu);
    }
    // up right
    if (has_up(bin_id) && has_right(bin_id)) {
        loop(parts, tid, bin_id - bins_per_row + 1, heads_gpu, linked_list_gpu);
    }
    // right
    if (has_right(bin_id)) {
        loop(parts, tid, bin_id + 1, heads_gpu, linked_list_gpu);
    }
    // down right
    if (has_down(bin_id) && has_right(bin_id)) {
        loop(parts, tid, bin_id + bins_per_row + 1, heads_gpu, linked_list_gpu);
    }
    // down
    if (has_down(bin_id)) {
        loop(parts, tid, bin_id + bins_per_row, heads_gpu, linked_list_gpu);
    }
    // down left
    if (has_down(bin_id) && has_left(bin_id)) {
        loop(parts, tid, bin_id + bins_per_row - 1, heads_gpu, linked_list_gpu);
    }
    // left
    if (has_left(bin_id)) {
        loop(parts, tid, bin_id - 1, heads_gpu, linked_list_gpu);
    }
    // up left
    if (has_up(bin_id) && has_left(bin_id)) {
        loop(parts, tid, bin_id - bins_per_row - 1, heads_gpu, linked_list_gpu);
    }
}

// Integrate the ODE
__global__ void move_gpu(particle_t* parts, int num_parts, double size) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &parts[tid];

    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    // Bounce from walls
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
    // Initialize data objects needed later on
    // This function will be called once before the algorithm begins
    // No particle simulation here

    int host_bins_per_row = size / cutoff;
    int host_bin_count = host_bins_per_row * host_bins_per_row;
    double host_bin_size = size / host_bins_per_row;

    cudaMemcpyToSymbol(bins_per_row, &host_bins_per_row, sizeof(int));
    cudaMemcpyToSymbol(bin_count, &host_bin_count, sizeof(int));
    cudaMemcpyToSymbol(bin_size, &host_bin_size, sizeof(double));

    cudaMalloc((void**)&heads, host_bin_count * sizeof(int));
    cudaMalloc((void**)&linked_list, num_parts * sizeof(int));

    int blockSize = NUM_THREADS;

//    printf("NumParts:%d\n", num_parts);
//    printf("BinCount.%d\n", host_bin_count);
//    printf("BlockSize:%d\n", blockSize);
//    printf("NumBlocks:%d\n", numBlocks);
//    printf("Rebin NumBlocks:%d\n", rebin_numBlocks);
//    printf("TotalRebinThreads:%d\n", blockSize * rebin_numBlocks);

    numBlocks = (num_parts + blockSize - 1) / blockSize;
    rebin_numBlocks = (host_bin_count + blockSize - 1) / blockSize;
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Rebin
    rebin_gpu<<<rebin_numBlocks, NUM_THREADS>>>(parts, num_parts,
                                                heads, linked_list);
    // Compute forces
    compute_forces_gpu<<<numBlocks, NUM_THREADS>>>(parts, num_parts,
                                                   heads, linked_list);
    // Move particles
    move_gpu<<<numBlocks, NUM_THREADS>>>(parts, num_parts, size);
}