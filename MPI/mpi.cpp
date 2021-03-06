#include "common.h"
#include <mpi.h>
#include <cmath>
#include <unistd.h>
#include <algorithm>
#include <vector>
#include <set>
#include <iostream>

using namespace std;
// Put any static global variables here that you will use throughout the simulation.
typedef vector<particle_t*> bin_t;

double size;
int bin_row_count;
int bin_count;
double bin_size;
bin_t* bins;
vector<int> local_binID;
int* migrate_size;
int* disp_size;
int proc_count;
int num_of_procs_with_more_rows;
int rows_per_proc;



int inline get_bin_id(particle_t& particle) {
    int x, y;
    y = particle.y / bin_size;
    x = particle.x / bin_size;
    if (x == bin_row_count) {
        x--;
    }
    if (y == bin_row_count) {
        y--;
    }
    return y * bin_row_count + x;
}

int inline get_proc_id(int bin_id) {
    int row_id = bin_id / bin_row_count;
    if (row_id >= num_of_procs_with_more_rows * (rows_per_proc + 1)) {
        return (row_id - num_of_procs_with_more_rows * (rows_per_proc + 1)) / rows_per_proc + num_of_procs_with_more_rows;
    } else {
        return row_id / (rows_per_proc + 1);
    }
}

void assign_bins(particle_t* parts, int num_parts, int rank) {
    for (int i = 0; i < num_parts; i++) {
        int bin_id = get_bin_id(parts[i]);
        int proc_id = get_proc_id(bin_id);
        if (proc_id == rank) {
            bins[bin_id].push_back(&parts[i]);
        }
    }
}

void rebin(particle_t* original_parts, int num_parts, int rank, int num_procs) {
    vector<particle_t> particles_to_send;
    for (int binID: local_binID) {
        /*
        for (particle_t* p: bins[binID]) {
            int new_binID = get_bin_id(*p);
            int new_procID = get_proc_id(new_binID);
            if (new_procID != rank) {
                bins[binID].erase(remove(bins[binID].begin(), bins[binID].end(), p), bins[binID].end());
                particles_to_send.push_back(*p);
            } else if(new_binID != binID) {
                bins[binID].erase(remove(bins[binID].begin(), bins[binID].end(), p), bins[binID].end());
                bins[new_binID].push_back(p);
            }
        }
        */
        for (auto it = bins[binID].begin(); it != bins[binID].end(); it++) {
            int new_binID = get_bin_id(**it);
            int new_procID = get_proc_id(new_binID);
            if (new_procID != rank) {
                particles_to_send.push_back(**it);
                bins[binID].erase(it--);
            } else if(new_binID != binID) {
                bins[new_binID].push_back(*it);
                bins[binID].erase(it--);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int particles_to_send_size = particles_to_send.size();
    MPI_Allgather(&particles_to_send_size, 1, MPI_INT, migrate_size, 1, MPI_INT, MPI_COMM_WORLD);

    disp_size[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        disp_size[i] = disp_size[i-1] + migrate_size[i-1];
    }

    particle_t* particles = new particle_t[num_parts];
    MPI_Allgatherv(&particles_to_send[0], particles_to_send.size(), PARTICLE, particles, migrate_size, disp_size, PARTICLE, MPI_COMM_WORLD);

    int overall_migrate_num = 0;
    for (int i = 0; i < num_procs; i++) {
        overall_migrate_num += migrate_size[i];
    }

    for (int i = 0; i < overall_migrate_num; i++) {
        int bin_id = get_bin_id(particles[i]);
        int proc_id = get_proc_id(bin_id);
        if (proc_id == rank) {
            particle_t particle_copy = particles[i];
            bins[bin_id].push_back(&original_parts[particle_copy.id - 1]);
            original_parts[particle_copy.id - 1].x = particle_copy.x;
            original_parts[particle_copy.id - 1].y = particle_copy.y;
            original_parts[particle_copy.id - 1].vx = particle_copy.vx;
            original_parts[particle_copy.id - 1].vy = particle_copy.vy;
            original_parts[particle_copy.id - 1].ax = particle_copy.ax;
            original_parts[particle_copy.id - 1].ay = particle_copy.ay;
        }
    }
    delete[] particles;
}
int max_partitions(int process_count) {
    for (int i = fmin(process_count, bin_count); i >= 1; i--) {
        if (bin_count % i == 0) {
            return i;
        }
    }
}

void get_local_binID(int rank) {
    if (rank < num_of_procs_with_more_rows) {
        for (int i = 0; i < rows_per_proc + 1; i++) {
            int row = rank * (rows_per_proc + 1) + i;
            for (int j = 0; j < bin_row_count; j++) {
                local_binID.push_back(row * bin_row_count + j);
            }
        }
    } else {
        for (int i = 0; i < rows_per_proc; i++) {
            int row = num_of_procs_with_more_rows * (rows_per_proc + 1) + (rank - num_of_procs_with_more_rows) * rows_per_proc + i;
            for (int j = 0; j < bin_row_count; j++) {
                local_binID.push_back(row * bin_row_count + j);
            }
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size_, int rank, int num_procs) {

    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // Do not do any particle simulation here
    size = size_;
    bin_row_count = size / cutoff;
    bin_count = bin_row_count * bin_row_count;
    bin_size = size / bin_row_count;
    bins = new bin_t[bin_row_count * bin_row_count];
    proc_count = min(num_procs, bin_row_count);
    rows_per_proc = bin_row_count / proc_count;
    num_of_procs_with_more_rows = bin_row_count % proc_count;

    // assign bins
    if (rank < proc_count) {
        get_local_binID(rank);
        assign_bins(parts, num_parts, rank);
    }
    // init particles
    migrate_size = (int*) malloc(num_procs * sizeof(int));
    disp_size = (int*) malloc(num_procs * sizeof(int));
}

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}


bool inline has_up_bin(int bin_id) {
    return bin_id - bin_row_count > -1;
}
bool inline has_down_bin(int bin_id) {
    return bin_id + bin_row_count < bin_count;
}
bool inline has_left_bin(int bin_id) {
    return bin_id % bin_row_count != 0;
}
bool inline has_right_bin(int bin_id) {
    return bin_id % bin_row_count != bin_row_count - 1;
}

bool inline has_up_proc(int proc_id) {
    if (proc_id >= proc_count) {
        return false;
    }
    return proc_id > 0;
}
int inline get_up_proc(int proc_id) {
    return proc_id - 1;
}
bool inline has_down_proc(int proc_id) {
    return proc_id < proc_count - 1;
}
int inline get_down_proc(int proc_id) {
    return proc_id + 1;
}

vector<int>* get_up_border_bin_ids(int rank) {
    vector<int>* res = new vector<int>;
    int row;
    if (rank < num_of_procs_with_more_rows) {
        row = rank * (rows_per_proc + 1);
    } else {
        row = num_of_procs_with_more_rows * (rows_per_proc + 1) + (rank - num_of_procs_with_more_rows) * rows_per_proc;
    }
    for (int i = 0; i < bin_row_count; i++) {
        res->push_back(row * bin_row_count + i);
    }

    return res;
}
vector<int>* get_down_border_bin_ids(int rank) {
    vector<int>* res = new vector<int>();
    int row;
    if (rank < num_of_procs_with_more_rows) {
        row = rank * (rows_per_proc + 1);
        row += rows_per_proc;
    } else {
        row = num_of_procs_with_more_rows * (rows_per_proc + 1) + (rank - num_of_procs_with_more_rows) * rows_per_proc;
        row += rows_per_proc - 1;
    }
    for (int i = 0; i < bin_row_count; i++) {
        res->push_back(row * bin_row_count + i);
    }
    return res;
}

// direction: 0 --> up; 1 --> down
vector<particle_t>* send_particles(int rank, int direction, vector<MPI_Request*>* requests) {
    vector<particle_t>* to_send = new vector<particle_t>();
    vector<int>* border;
    if (direction == 0) {
        border = get_up_border_bin_ids(rank);
    } else {
        border = get_down_border_bin_ids(rank);
    }
    for (auto bin_id : *border) {
        for (auto part : bins[bin_id]) {
            to_send->push_back(*part);
        }
    }
    delete border;
    MPI_Request* request = new MPI_Request();
    requests->push_back(request);
    if (direction == 0) {
        MPI_Isend(&(*to_send)[0], to_send->size(), PARTICLE, get_up_proc(rank), 0, MPI_COMM_WORLD, request);
    } else {
        MPI_Isend(&(*to_send)[0], to_send->size(), PARTICLE, get_down_proc(rank), 0, MPI_COMM_WORLD, request);
    }
    return to_send;
}
// direction: 0 --> up; 1 --> down
particle_t* receive_paritcles(int num_parts, int rank, int direction, set<int>* surrounding_bin_ids) {
    MPI_Status status;
    particle_t* recv_buff = new particle_t[5 * bin_row_count];
    if (direction == 0) {
        MPI_Recv(recv_buff, num_parts, PARTICLE, get_up_proc(rank), 0, MPI_COMM_WORLD, &status);
    } else {
        MPI_Recv(recv_buff, num_parts, PARTICLE, get_down_proc(rank), 0, MPI_COMM_WORLD, &status);
    }

    int count;
    MPI_Get_count(&status, PARTICLE, &count);
    for (int i = 0; i < count; i++) {
        int bin_id = get_bin_id(recv_buff[i]);
        bins[bin_id].push_back(&recv_buff[i]);
        surrounding_bin_ids->insert(bin_id);
    }
    return recv_buff;
}

void inline loop(particle_t* part, int another_bin_id) {
    for (particle_t* neighbor : bins[another_bin_id]) {
        apply_force(*part, *neighbor);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    vector<MPI_Request*> requests;
    vector<vector<particle_t>*> send_buffers;
    if (has_up_proc(rank)) {
        send_buffers.push_back(send_particles(rank, 0, &requests));
    }
    if (has_down_proc(rank)) {
        send_buffers.push_back(send_particles(rank, 4, &requests));
    }
    vector<particle_t*> recv_buffers;
    set<int> surrounding_bin_ids;
    if (has_up_proc(rank)) {
        recv_buffers.push_back(receive_paritcles(num_parts, rank, 0, &surrounding_bin_ids));
    }
    if (has_down_proc(rank)) {
        recv_buffers.push_back(receive_paritcles(num_parts, rank, 4, &surrounding_bin_ids));
    }

    // calculate force here
    for (auto bin_id : local_binID) {
        for (auto part : bins[bin_id]) {
            part->ax = part->ay = 0;
            loop(part, bin_id);
            if (has_up_bin(bin_id)) {
                loop(part, bin_id - bin_row_count);
            }
            if (has_up_bin(bin_id) && has_right_bin(bin_id)) {
                loop(part, bin_id - bin_row_count + 1);
            }
            if (has_right_bin(bin_id)) {
                loop(part, bin_id + 1);
            }
            if (has_down_bin(bin_id) && has_right_bin(bin_id)) {
                loop(part, bin_id + bin_row_count + 1);
            }
            if (has_down_bin(bin_id)) {
                loop(part, bin_id + bin_row_count);
            }
            if (has_down_bin(bin_id) && has_left_bin(bin_id)) {
                loop(part, bin_id + bin_row_count - 1);
            }
            if (has_left_bin(bin_id)) {
                loop(part, bin_id - 1);
            }
            if (has_up_bin(bin_id) && has_left_bin(bin_id)) {
                loop(part, bin_id - bin_row_count - 1);
            }
        }
    }
    MPI_Status array_of_statuses[requests.size()];
    for (auto request : requests) {
        MPI_Status status;
        MPI_Wait(request, &status);
        delete request;
    }
    for (int i = 0; i < send_buffers.size(); i++) {
        delete send_buffers[i];
    }

    for (auto bin_id : local_binID) {
        for (auto part : bins[bin_id]) {
            move(*part, size);
        }
    }

    // end calculate

    for (auto bin_id : surrounding_bin_ids) {
        bins[bin_id].clear();
    }
    for (int i = 0; i < recv_buffers.size(); i++) {
        delete[] recv_buffers[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    rebin(parts, num_parts, rank, num_procs);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    vector<particle_t> local_particles;
    for (int binID: local_binID) {
        for (particle_t* p: bins[binID]) {
            local_particles.push_back(*p);
        }
    }
    int* gather_particles_size = new int[num_procs];
    int* gather_disp_size = new int[num_procs];

    int error_code = 0;

    int local_particles_size = local_particles.size();
    error_code = MPI_Gather(&local_particles_size, 1, MPI_INT, gather_particles_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (error_code != 0) {
        cout << "MPI_Gather error!!" << endl;
    }
    gather_disp_size[0] = 0;
    for (int i = 1; i < num_procs; i++) {
        gather_disp_size[i] = gather_disp_size[i-1] + gather_particles_size[i-1];
    }

    //particle_t recv_buf[num_parts];
    particle_t* recv_buf = new particle_t[num_parts];
    error_code = MPI_Gatherv(&local_particles[0], local_particles.size(), PARTICLE, recv_buf,
                             gather_particles_size, gather_disp_size, PARTICLE, 0, MPI_COMM_WORLD);

    if (error_code != 0) {
        cout << "MPI_Gatherv error!!" << endl;
    }

    if (rank == 0) {
        for (int i = 0; i < num_parts; i++) {
            particle_t p = recv_buf[i];
            parts[p.id-1].x = p.x;
            parts[p.id-1].y = p.y;
            parts[p.id-1].ax = p.ax;
            parts[p.id-1].ay = p.ay;
            parts[p.id-1].vx = p.vx;
            parts[p.id-1].vy = p.vy;
        }
    }
    delete[] recv_buf;
    delete[] gather_particles_size;
    delete[] gather_disp_size;
}
