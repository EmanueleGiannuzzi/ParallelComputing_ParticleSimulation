#include "common.h"
#include <mpi.h>
#include <cmath>
#include <vector>

using namespace std;

int banana = 0;

int cose = 0;

struct bin_t {
    int id;
    vector<particle_t*> particles;

    bin_t() {}

    bin_t(int _id){
        init(_id);
    }

    void init(int _id) {
        id = _id;
    }

    void add_particle(particle_t* particle) {
        particles.push_back(particle);

        cose++;
    }

    void clear() {
        particles.clear();
    }

    unsigned long size() const{
        return particles.size();
    }
};

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

struct focus {
    bin_t* focus_bin;
    bin_t** neighbours;
    int neighbours_size;

    focus(bin_t* _focus, int _neighbours_size) {
        init(_focus, neighbours_size);
    }

    focus(){}

    void init(bin_t* _focus, int _neighbours_size) {
        focus_bin = _focus;
        neighbours_size = _neighbours_size;
        neighbours = new bin_t*[_neighbours_size];
    }

    int id() const{
        return focus_bin->id;
    }

    void apply_forces(particle_t* parts, int particle_count) {
        for(particle_t* focus_particle : focus_bin->particles){
            focus_particle->ax = 0;
            focus_particle->ay = 0;

            banana++;
            for (int j = 0; j < particle_count; ++j) {
                apply_force(*focus_particle, parts[j]);
            }

//            for(int i = 0; i < neighbours_size; ++i) {
//                bin_t* neighbour_bin = neighbours[i];
//
//                for(particle_t* neighbour_particle : neighbour_bin->particles){
//                    apply_force(*focus_particle, *neighbour_particle);
//                }
//            }
        }
    }

    void move_particles(double size) {
        for(particle_t* focus_particle : focus_bin->particles){
            move(*focus_particle, size);
        }
    }
};

double bin_size;
int bin_count;
int bin_row_count;
int bin_per_proc;
focus* focuses; //size: focus_count
int focus_count;
map<int, bin_t> bin_data;

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
        local_b_ids[i] = alternate_id;
    }
    return local_b_ids;
}

vector<int>& get_bin_neighbours_id(int focus_bin_id) {
    vector<int>* neighbours = new vector<int>();
    for(int i = 0; i<directions.size(); ++i) {
        if(directions_check[i](focus_bin_id)){
            neighbours->push_back(directions[i](focus_bin_id));
        }
    }
    return *neighbours;
}

bin_t* get_bin(int id, bool create_new = true) {
    if(bin_data.count(id) > 0) {
        return &bin_data[id];
    }
    else if (create_new) {
        bin_t* bin_ptr = new bin_t(id);
        bin_data.emplace(id, *bin_ptr);
        return bin_ptr;
    }
    return nullptr;
}

void init_focuses(int rank, int num_procs) {
    int *focus_ids = get_focus_ids(rank, num_procs);
    vector<int> neighbour_ids[focus_count];

    for (int i = 0; i < focus_count; ++i) {
        int focus_id = focus_ids[i];
        neighbour_ids[i] = get_bin_neighbours_id(focus_id);
    }

    focuses = new focus[focus_count];
    for (int i = 0; i < focus_count; ++i) {
        focus *focus = &focuses[i];

        bin_t *focus_bin = get_bin(focus_ids[i]);
        int neighbours_size = (int) neighbour_ids[i].size();
        focus->init(focus_bin, neighbours_size);

        for (int j = 0; j < neighbours_size; ++j) {
            bin_t *neighbour_bin = get_bin(neighbour_ids[i].at(j));
            focus->neighbours[j] = neighbour_bin;
        }
    }
}

int get_bin_id(const particle_t& particle) {
    int x, y;
    y = (int)(particle.y / bin_size);
    x = (int)(particle.x / bin_size);
    return y * bin_row_count + x;
}

void binning(particle_t* parts, int particle_count) {
    for (auto& kv : bin_data) {
        kv.second.clear();
    }
    for(int i = 0; i < particle_count; ++i) {
        particle_t* particle = &parts[i];
        int particle_bin_id = get_bin_id(*particle);
        bin_t* particle_bin = get_bin(particle_bin_id, false);
        if(particle_bin != nullptr) {
            particle_bin->add_particle(particle);
        }
        else {
            string errorMessage = string("Particle ") + to_string(particle->id) + "not added to bin " + to_string(particle_bin_id);
            throw runtime_error(errorMessage);
        }
    }

    printf("COSE3 %d\n", cose);
    cose = 0;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    calculate_grid_parameters(size, num_procs);
    init_focuses(rank, num_procs);
    binning(parts, num_parts);
}

void simulate_focuses(double size, particle_t* parts, int particle_count) {
    for(int i = 0; i < focus_count; ++i)  {
        focuses[i].apply_forces(parts, particle_count);
    }

    for(int i = 0; i < focus_count; ++i)  {
        focuses[i].move_particles(size);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    simulate_focuses(size, parts, num_parts);
    binning(parts, num_parts);


//    int count = 0;
//    for (auto& kv : bin_data) {
//        count += kv.second.size();
//    }
    int count = 0;
    for(int i = 0; i < focus_count; ++i)  {
        count += focuses[i].focus_bin->size();
    }
    printf("BANANA %d %d \n", banana, count);
    banana = 0;
//    printf("-----------RANK %d Done simulating\n", rank);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

}