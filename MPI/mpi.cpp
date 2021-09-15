#include "common.h"
#include <cmath>
#include <vector>

using namespace std;

//region Phisics
void apply_force(particle_t& particle, particle_t& neighbor) {
//    printf("P %lu -> %lu\n", particle.id, neighbor.id);
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

//    if(particle.id != neighbor.id) {
//        printf("P %lu -> %lu [%f]\n", particle.id, neighbor.id, coef);
//    }
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

//    printf("Particle %lu %f %f %f %f %f %f\n", p.id, p.x, p.y, p.vx, p.vy, p.ax, p.ay);
}
//endregion

struct bin_t {
    int id;
    vector<particle_t*> particles;
    vector<bin_t*> neighbours;

    explicit bin_t(int _id){
        id = _id;
    }

    void add_particle(particle_t* particle) {
        particles.push_back(particle);
    }

    void clear() {
        particles.clear();
    }

    unsigned long size() const{
        return particles.size();
    }

    void apply_forces() {
        for(particle_t* focus_particle : particles){
            focus_particle->ax = 0;
            focus_particle->ay = 0;
            for(particle_t* neighbour_particle : particles){
                apply_force(*focus_particle, *neighbour_particle);
            }
            for(int i = 0; i < neighbours.size(); ++i) {
                for(particle_t* neighbour_particle : neighbours[i]->particles){
                    apply_force(*focus_particle, *neighbour_particle);
                }
            }
        }
    }

    void move_particles(double size) {
        for(particle_t* focus_particle : particles) {
            move(*focus_particle, size);
        }
    }
};

double bin_size;
int bin_count;
int bin_row_count;
int bin_per_proc;
int* focus_ids;
int focus_count;
vector<bin_t> bin_data;

//region Partitioning
typedef int (*direction_id_func)(int);
vector<direction_id_func> directions({
                                             [](int bin_id) {//0 - TOP
                                                 return bin_id - bin_row_count;
                                             },
                                             [](int bin_id) {//1 - BOTTOM
                                                 return bin_id + bin_row_count;
                                             },
                                             [](int bin_id) {//2 - LEFT
                                                 return bin_id - 1;
                                             },
                                             [](int bin_id) {//3 - RIGHT
                                                 return bin_id + 1;
                                             },
                                             [](int bin_id) {//4 - TOPLEFT
                                                 return bin_id - bin_row_count - 1;
                                             },
                                             [](int bin_id) {//5 - TOPRIGHT
                                                 return bin_id - bin_row_count + 1;
                                             },
                                             [](int bin_id) {//6 - BOTTOMLEFT
                                                 return bin_id + bin_row_count - 1;
                                             },
                                             [](int bin_id) {//7 - BOTTOMRIGHT
                                                 return bin_id + bin_row_count + 1;
                                             }
                                     });
typedef bool (*direction_check_func)(int);
vector<direction_check_func> directions_check({
                                                      [](int bin_id) {//0 - TOP
                                                          return bin_id - bin_row_count > -1;
                                                      },
                                                      [](int bin_id) {//1 - BOTTOM
                                                          return bin_id + bin_row_count < bin_count;
                                                      },
                                                      [](int bin_id) {//2 - LEFT
                                                          return bin_id % bin_row_count != 0;
                                                      },
                                                      [](int bin_id) {//3 - RIGHT
                                                          return bin_id % bin_row_count != bin_row_count - 1;
                                                      },
                                                      [](int bin_id) {//4 - TOPLEFT
                                                          return (bin_id - bin_row_count > -1) && (bin_id % bin_row_count != 0);//TOP && LEFT
                                                      },
                                                      [](int bin_id) {//5 - TOPRIGHT
                                                          return (bin_id - bin_row_count > -1) && (bin_id % bin_row_count != bin_row_count - 1);//TOP && RIGHT
                                                      },
                                                      [](int bin_id) {//6 - BOTTOMLEFT
                                                          return (bin_id + bin_row_count < bin_count) && (bin_id % bin_row_count != 0);//BOTTOM && LEFT
                                                      },
                                                      [](int bin_id) {//7 - BOTTOMRIGHT
                                                          return (bin_id + bin_row_count < bin_count) && (bin_id % bin_row_count != bin_row_count - 1);//BOTTOM && RIGHT
                                                      }
                                              });
//endregion

void calculate_grid_parameters(double size, int num_procs) {
    bin_row_count = (int)floor(size / cutoff);
    bin_size = size / bin_row_count;

    printf("SIZE %f - BIN SIZE %f - CUTOFF %f - BIN_ROW_COUNT %d\n", size, bin_size, cutoff, bin_row_count);
    bin_count = bin_row_count * bin_row_count;

    if(bin_count > num_procs) {
        bin_per_proc = (int)(bin_count / num_procs);
    }
    else {
        bin_per_proc = 1;
    }
}

int* get_focus_ids(int rank, int num_procs) {
    focus_count = bin_per_proc;
    int extra_count = bin_count % num_procs;
    bool extra = extra_count != 0;
    if(rank == 0 && extra) {
        focus_count += extra_count;
    }
    if(bin_per_proc == 1 && rank >= bin_count) {
        focus_count = 0;
    }
    int* local_b_ids = new int[focus_count];

    int start = rank * bin_per_proc + (rank == 0 ? 0 : (extra ? extra_count : 0));
    for(int i = 0 ; i < focus_count ; i++ ) {
        int alternate_id = start + i;//TODO: normal_to_alternate_sqmatrix_id https://github.com/EmanueleGiannuzzi/ParallelComputing_ParticleSimulation/blob/98b553815fb5b516d3e0519c6e21bf0bb3f1623d/MPI/mpi.cpp#L248
        //int alternate_id = normal_to_alternate_sqmatrix_id(start + i, bin_row_count);
        local_b_ids[i] = alternate_id;
    }
    return local_b_ids;
}

vector<int> get_bin_neighbours_ids(int focus_bin_id) {
    vector<int> neighbours;
    for(int i = 0; i<directions.size(); ++i) {
        if(directions_check[i](focus_bin_id)){
            neighbours.push_back(directions[i](focus_bin_id));
        }
    }
    return neighbours;
}

bin_t* get_bin(int id) {
    return &bin_data.at(id);
}

void init_focuses(int rank, int num_procs) {
    focus_ids = get_focus_ids(rank, num_procs);
    for (int bin_id = 0; bin_id < bin_count; ++bin_id) {
        bin_t bin(bin_id);
        bin_data.insert(bin_data.begin() + bin.id, bin);
    }

    for (int bin_id = 0; bin_id < bin_count; ++bin_id) {
        bin_t* bin = get_bin(bin_id);
        vector<int> neighbour_ids = get_bin_neighbours_ids(bin_id);

        int neighbours_size = (int) neighbour_ids.size();

        for (int j = 0; j < neighbours_size; ++j) {
            bin_t *neighbour_bin = get_bin(neighbour_ids.at(j));

            bin->neighbours.push_back(neighbour_bin);
        }
    }
}

int get_bin_id(const particle_t* particle) {
    int x, y;
    y = (int)(particle->y / bin_size);
    x = (int)(particle->x / bin_size);
    return y * bin_row_count + x;
}

void binning(particle_t* parts, int num_parts) {
    for (bin_t& bin : bin_data) {
        bin.clear();
    }

    for(int i = 0; i < num_parts; ++i) {
        particle_t* particle = &parts[i];
        int particle_bin_id = get_bin_id(particle);

        bin_t* particle_bin = get_bin(particle_bin_id);
        if(particle_bin != nullptr) {
            particle_bin->add_particle(particle);
        }
        else {
            string errorMessage = string("Particle ") + to_string(particle->id) + " not added to bin " + to_string(particle_bin_id);
            throw runtime_error(errorMessage);
        }
    }
}

bin_t* get_focus(int focus_id) {
    return get_bin(focus_ids[focus_id]);
}

//region Communications
int get_rank_from_bin_id(int bin_id, int num_procs) {
    int extra_count = bin_count % num_procs;
    bool extra = extra_count != 0;
    if(extra) {
        int rank0_bin_count = bin_per_proc + extra_count;
        if(bin_id < rank0_bin_count) {
            return 0;
        }
        else {
            return ((bin_id - rank0_bin_count ) / bin_per_proc) + 1;
        }
    }
    else {
        return (int)(bin_id / bin_per_proc);
    }

}

particle_t* serialize_bin_data(const bin_t* bin) {
    unsigned long buffer_size = bin->size();

    particle_t* buffer = (particle_t*)malloc(buffer_size * sizeof(particle_t));
    for(int i = 0; i < buffer_size; ++i) {
        memcpy(&buffer[i], bin->particles[i], sizeof(particle_t));
    }
    return buffer;
}

void deserialize_bin_data(particle_t buffer[], int size, particle_t* parts, int parts_size) {
    const particle_t* pointer = buffer;
    for(int i = 0; i < size; ++i) {
        particle_t particle;
        memcpy(&particle, pointer, sizeof(particle_t));
        pointer++;
        for(int j = 0; j < parts_size; ++j) {//TODO: Maybe use a map?
            if(parts[j].id == particle.id) {
                memcpy(&parts[j], &particle, sizeof(particle_t));
                break;
            }
        }
    }
}

void start_send_to_neighbours(vector<MPI_Request*>& requests, vector<particle_t*>& send_buffers, int my_rank, int num_procs) {
    for(int i = 0; i < focus_count; ++i)  {
        const bin_t* focus = get_focus(i);
        particle_t* buffer = serialize_bin_data(focus);
        unsigned long buffer_size = focus->size();

        MPI_Request* request;
        for(int dir_id = 0; dir_id < directions.size(); ++dir_id) {
            if(directions_check[dir_id](focus->id)) {
                int neighbour_bin_id = directions[dir_id](focus->id);
                int dest_rank = get_rank_from_bin_id(neighbour_bin_id, num_procs);

                if(dest_rank != my_rank) {
                    request = new MPI_Request();
                    MPI_Isend(buffer, (int)buffer_size, PARTICLE, dest_rank, 0, MPI_COMM_WORLD, request);
//                    printf(">Rank %d sent to %d, %d particles\n", my_rank, dest_rank, (int)buffer_size);
                    requests.push_back(request);
                }
            }
        }
        send_buffers.push_back(buffer);
    }
}

void wait_and_clear_buffer(vector<MPI_Request*>& requests, vector<particle_t*>& buffers) {
    if(requests.empty() || buffers.empty()) {
        return;
    }
    MPI_Request arr[requests.size()];
    for(int i = 0; i < requests.size(); ++i) {
        arr[i] = *requests[i];
    }

    MPI_Waitall((int) requests.size(), arr, MPI_STATUSES_IGNORE);
    for(MPI_Request* request : requests) {
        delete request;
    }
    for(particle_t* buffer : buffers) {
        free(buffer);
    }
}

void clear_requests_ready_only(vector<MPI_Request*>& requests, vector<particle_t*>& buffers) {
    int index, flag;
    do {
        MPI_Testany((int) requests.size(), requests.front(), &index, &flag, MPI_STATUS_IGNORE);
        if(flag) {
            free(buffers[index]);
            buffers.erase(buffers.begin() + index);
            MPI_Request_free(*(requests.begin() + index));
            requests.erase(requests.begin() + index);

        }
    } while (flag);
}

bool is_my_focus(int bin_id) {
    for(int i = 0; i < focus_count; ++i) {
        if(focus_ids[i] == bin_id){
            return true;
        }
    }
    return false;
}

int get_receive_count() {
    int count = 0;
    for(int i = 0; i < focus_count; ++i) {
        const bin_t* focus = get_focus(i);
        for(auto neighbour : focus->neighbours) {
            if(!is_my_focus(neighbour->id)) {
                count++;
            }
        }
    }
    return count;
}


void receive_from_neighbours(particle_t* parts, int particle_count, int my_rank) {
    static int max_receive_count = get_receive_count();//TODO: Static?
    printf("[myrank:%d] RECEIVE COUNT %d\n", my_rank, max_receive_count);

    int receive_count = 0;
    while(receive_count < max_receive_count) {
        printf("rank%d - %d\n", my_rank, receive_count);
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int buffer_size = 0;
        MPI_Get_count(&status, PARTICLE, &buffer_size);
        int source_rank = status.MPI_SOURCE;

//        printf("Rank %d receiving from %d, %d particles\n", my_rank, source_rank, buffer_size);
        particle_t* buffer = (particle_t*)malloc(buffer_size * sizeof(particle_t));
        MPI_Recv(buffer, buffer_size, PARTICLE, source_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        deserialize_bin_data(buffer, buffer_size, parts, particle_count);
        receive_count++;
    }
//    printf("[myrank:%d] Received: %d\n", my_rank, receive_count);
}
//endregion

void simulate_focuses(double size) {
    for(int i = 0; i < focus_count; ++i)  {
        get_focus(i)->apply_forces();
    }

    for(int i = 0; i < focus_count; ++i)  {
        get_focus(i)->move_particles(size);
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    calculate_grid_parameters(size, num_procs);
    init_focuses(rank, num_procs);
//    binning(parts, num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    binning(parts, num_parts);
    simulate_focuses(size);
    printf("-----------RANK %d Done simulating\n", rank);

    vector<MPI_Request*> send_requests;
    vector<particle_t*> send_buffers;
    start_send_to_neighbours(send_requests, send_buffers, rank, num_procs);
//    clear_requests_ready_only(send_requests, send_buffers);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("-----------RANK %d Done sending\n", rank);

    receive_from_neighbours(parts, num_parts, rank);//Blocking

    printf("-----------RANK %d Done receive\n", rank);

//    clear_requests_ready_only(send_requests, send_buffers);


    wait_and_clear_buffer(send_requests, send_buffers);

    printf("Barrier reached %d\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    if(rank != 0) {
        vector<particle_t> buffer;
        for(int i = 0; i < focus_count; ++i)  {
            const bin_t* focus = get_focus(i);
            particle_t* focus_buffer = serialize_bin_data(focus);
            unsigned long focus_particle_count = focus->size();

            for(int j = 0; j < focus_particle_count; ++j) {
                buffer.push_back(focus_buffer[j]);
            }
        }

        MPI_Send(&buffer.front(), (int)buffer.size(), PARTICLE, 0, 1, MPI_COMM_WORLD);
    }
    else {
        for(int i = 0; i < num_procs - 1; ++i) {
            particle_t* buffer;
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            int source_rank = status.MPI_SOURCE;
            int buffer_size;
            MPI_Get_count(&status, PARTICLE, &buffer_size);
            buffer = (particle_t*)malloc(buffer_size * sizeof(particle_t));
            MPI_Recv(buffer, buffer_size, PARTICLE, source_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            deserialize_bin_data(buffer, buffer_size, parts, num_parts);

            delete buffer;
        }
    }
}