#include "common.h"
#include <mpi.h>
#include <math.h>

double bin_size;
int bin_row_count;
int bin_per_proc;

int* local_bin_ids;

// Put any static global variables here that you will use throughout the simulation.

int normal_to_alternate_sqmatrix_id(int id, int size) {
    bool leftToRight = ((int)(id / size)) % 2 == 0;

    if(leftToRight) {
        return id;
    }
    else{
        return (size - 1 - (id % size)) + (floor(id / size) * size);
    }
}

int* get_local_bin_ids(int rank, int _bin_per_proc, int _bin_count, int _bin_row_count) {
    int local_bin_count = _bin_per_proc;
    bool extra = _bin_count % _bin_per_proc != 0;
    if(rank == 0 && extra) {
        local_bin_count++;
    }
    int* local_b_ids = new int[local_bin_count];

    int start = rank * _bin_per_proc + (rank == 0 ? 0 : (extra ? 1 : 0));
    for(int i = 0 ; i < local_bin_count ; i++ ) {
        int alternate_id = normal_to_alternate_sqmatrix_id(start + i, _bin_row_count);
        local_b_ids[i] = alternate_id;
    }
    return local_b_ids;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    bin_size = size / cutoff;
    bin_row_count = floor(bin_size);
    int bin_count = bin_row_count * bin_row_count;

    if(bin_count > num_procs) {
        bin_per_proc = (int)(bin_count / num_procs);
    }
    else {
        bin_per_proc = 1;
    }
    local_bin_ids = get_local_bin_ids(rank, bin_per_proc, bin_count, bin_row_count);

//    if(rank == 0)
//    printf("BANANA: %d\n", bin_row_count);
//    int local_bin_count = bin_per_proc;
//    bool extra = bin_count % bin_per_proc != 0;
//    if(rank == 0 && extra) {
//        local_bin_count++;
//    }
//    printf("%d - %d: ", rank, local_bin_count);
//    for(int i = 0; i < local_bin_count; i++) {
//        printf("%d, ", local_bin_ids[i]);
//    }
//    printf("\n");
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
}