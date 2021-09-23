#include "common.h"
#include <stdio.h>
#include <cuda.h>

#define NUM_THREADS 256

using namespace std;

// Put any static global variables here that you will use throughout the simulation.
int blks;
double bin_size;//TODO: Allocate on GPU
int bin_count;
int bin_row_count;
int* bin_map;
int** bin_neighbours_map;

//region Directions
__device__ bool inline has_up(int bin_id, int bins_per_row) {
    return bin_id - bins_per_row > -1;
}
__device__ bool inline has_down(int bin_id, int bins_per_row, int _bin_count) {
    return bin_id + bins_per_row < _bin_count;
}
__device__ bool inline has_left(int bin_id, int bins_per_row) {
    return bin_id % bins_per_row != 0;
}
__device__ bool inline has_right(int bin_id, int bins_per_row) {
    return bin_id % bins_per_row != bins_per_row - 1;
}
//endregion

void calculate_grid_parameters(double size) {
    bin_row_count = (int)floor(size / cutoff);
    bin_size = size / bin_row_count;

    printf("SIZE %f - BIN SIZE %f - CUTOFF %f - BIN_ROW_COUNT %d\n", size, bin_size, cutoff, bin_row_count);
    bin_count = bin_row_count * bin_row_count;
}

__global__ void init_bin_neighbours(int _bin_count, int** _bin_neighbours_map, int bins_per_row) {
    int bin_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (bin_id >= _bin_count)
        return;

    const int NEIGHBOURS_SIZE = 8;

    cudaMalloc((void**)&_bin_neighbours_map[bin_id], NEIGHBOURS_SIZE * sizeof(int));
    int count = 0;
    _bin_neighbours_map[bin_id] = new int[NEIGHBOURS_SIZE];
    if (has_up(bin_id, bins_per_row)) {
        _bin_neighbours_map[bin_id][count] = bin_id - bins_per_row;
        count++;
    }
    // up right
    if (has_up(bin_id, bins_per_row) && has_right(bin_id, bins_per_row)) {
        _bin_neighbours_map[bin_id][count] = bin_id - bins_per_row + 1;
        count++;
    }
    // right
    if (has_right(bin_id, bins_per_row)) {
        _bin_neighbours_map[bin_id][count] = bin_id + 1;
        count++;
    }
    // down right
    if (has_down(bin_id, bins_per_row, _bin_count) && has_right(bin_id, bins_per_row)) {
        _bin_neighbours_map[bin_id][count] = bin_id + bins_per_row + 1;
        count++;
    }
    // down
    if (has_down(bin_id, bins_per_row, _bin_count)) {
        _bin_neighbours_map[bin_id][count] = bin_id + bins_per_row;
        count++;
    }
    // down left
    if (has_down(bin_id, bins_per_row, _bin_count) && has_left(bin_id, bins_per_row)) {
        _bin_neighbours_map[bin_id][count] = bin_id + bins_per_row - 1;
        count++;
    }
    // left
    if (has_left(bin_id, bins_per_row)) {
        _bin_neighbours_map[bin_id][count] = bin_id - 1;
        count++;
    }
    // up left
    if (has_up(bin_id, bins_per_row) && has_left(bin_id, bins_per_row)) {
        _bin_neighbours_map[bin_id][count] = bin_id - bins_per_row - 1;
        count++;
    }

    for(; count < NEIGHBOURS_SIZE; ++count) {
        _bin_neighbours_map[bin_id][count] = -1;
    }
}

__device__ int get_bin_id(const particle_t& particle, double _bin_size, int _bin_row_count) {
    int x, y;
    y = (int)(particle.y / _bin_size);
    x = (int)(particle.x / _bin_size);
    return y * _bin_row_count + x;
}

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

__device__ bool find_bin_id(const int* array, int size, int element) {
    for(int i = 0; i < size; ++i) {
        if(array[i] == element) {
            return true;
        }
        else if(array[i] < 0) {
            return false;
        }
    }
    return false;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, const int* _bin_map, int** _bin_neighbours_map) {
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particles[tid].ax = particles[tid].ay = 0;

    int bin_id = _bin_map[tid];
    const int* neighbours = _bin_neighbours_map[bin_id];
    const int NEIGHBOURS_SIZE = 8;

    for (int neighbour_part_id = 0; neighbour_part_id < num_parts; neighbour_part_id++) {

//        printf("NEIGHBOUR %d\n", neighbour_part_id);
        int neighbour_bin_id = _bin_map[neighbour_part_id];
//        if(find_bin_id(neighbours, NEIGHBOURS_SIZE, neighbour_bin_id)) {
            apply_force_gpu(particles[tid], particles[neighbour_part_id]);

//            printf("Collision %d -> %d\n", tid, neighbour_part_id);
//        }
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

__global__ void rebin(particle_t* particles, double _bin_size, int _bin_row_count, int num_parts, int* bin_map) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    int bin_id = get_bin_id(particles[tid], _bin_size, _bin_row_count);
    bin_map[tid] = bin_id;
}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory
    // Do not do any particle simulation here

    calculate_grid_parameters(size);

    cudaMalloc((void**)&bin_neighbours_map, bin_count * sizeof(int*));
    int bin_blks = (bin_count + NUM_THREADS - 1) / NUM_THREADS;
    init_bin_neighbours<<<bin_blks, NUM_THREADS>>>(bin_count, bin_neighbours_map, bin_row_count);

    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    cudaMalloc((void**)&bin_map, num_parts * sizeof(int));
    rebin<<<blks, NUM_THREADS>>>(parts, bin_size, bin_row_count, num_parts, bin_map);

}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Compute forces

    printf("STEP\n");
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, bin_map, bin_neighbours_map);

    // Move particles
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);

    rebin<<<blks, NUM_THREADS>>>(parts, bin_size, bin_row_count, num_parts, bin_map);
}