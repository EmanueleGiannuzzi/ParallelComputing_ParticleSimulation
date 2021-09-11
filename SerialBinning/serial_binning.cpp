#include "common.h"
#include <cmath>
#include <vector>
#include <map>

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

vector<int>* get_bin_neighbours_ids(int focus_bin_id) {
    vector<int>* neighbours = new vector<int>();
    for(int i = 0; i<directions.size(); ++i) {
        if(directions_check[i](focus_bin_id)){
            neighbours->push_back(directions[i](focus_bin_id));
        }
    }
    return neighbours;
}

bin_t* get_bin(int id) {
    return &bin_data.at(id);
}

void init_focuses(int rank, int num_procs) {
    focus_ids = get_focus_ids(rank, num_procs);
    vector<int>* neighbour_ids[bin_count];

    for (int bin_id = 0; bin_id < bin_count; ++bin_id) {
        neighbour_ids[bin_id] = get_bin_neighbours_ids(bin_id);
    }

    for (int bin_id = 0; bin_id < bin_count; ++bin_id) {
        bin_t bin(bin_id);
        bin_data.insert(bin_data.begin() + bin.id, bin);
    }

    for (int bin_id = 0; bin_id < bin_count; ++bin_id) {
        bin_t* bin = get_bin(bin_id);

        int neighbours_size = (int) neighbour_ids[bin_id]->size();

        for (int j = 0; j < neighbours_size; ++j) {
            bin_t *neighbour_bin = get_bin(neighbour_ids[bin_id]->at(j));

            bin->neighbours.push_back(neighbour_bin);
        }
    }

    for (int bin_id = 0; bin_id < bin_count; ++bin_id) {
        delete neighbour_ids[bin_id];
    }
}

int get_bin_id(const particle_t& particle) {
    int x, y;
    y = (int)(particle.y / bin_size);
    x = (int)(particle.x / bin_size);
    return y * bin_row_count + x;
}

void binning(particle_t* parts, int particle_count) {
    for (bin_t& bin : bin_data) {
        bin.clear();
    }
    for(int i = 0; i < particle_count; ++i) {
        particle_t* particle = &parts[i];
        int particle_bin_id = get_bin_id(*particle);
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
    binning(parts, num_parts);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    simulate_focuses(size);
    binning(parts, num_parts);
}