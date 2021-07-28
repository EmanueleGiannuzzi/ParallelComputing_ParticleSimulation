#include "common.h"
#include <mpi.h>
#include <math.h>
#include <vector>
#include <map>

using namespace std;

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
    }

    void clear() {
        particles.clear();
    }
};

map<int, bin_t*> bin_data;

bin_t* get_bin(int id, bool create_new = true) {
    if(bin_data.count(id) > 0) {
        return bin_data[id];
    }
    else if (create_new){
        bin_t* bin_ptr = new bin_t(id);
        bin_data.emplace(id, bin_ptr);
        return bin_ptr;
    }
    return nullptr;
}

// Put any static global variables here that you will use throughout the simulation.

double bin_size;
int bin_count;
int bin_row_count;
int bin_per_proc;
int focus_count;

int* focus_ids;
vector<int>* neighbour_ids;


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

     int id() {
        return focus_bin->id;
    }
};

int inline get_bin_id(particle_t& particle) {
    int x, y;
    y = particle.y / bin_size;
    x = particle.x / bin_size;
    return y * bin_row_count + x;
}

struct local_bins {
    focus* focuses;

    static int size() {
        return focus_count;
    }
} bins;

int normal_to_alternate_sqmatrix_id(int id, int size) {
    bool leftToRight = ((int)(id / size)) % 2 == 0;

    if(leftToRight) {
        return id;
    }
    else{
        return (size - 1 - (id % size)) + (floor(id / size) * size);
    }
}

int* get_focus_ids(int rank) {
    focus_count = bin_per_proc;
    bool extra = bin_count % bin_per_proc != 0;
    if(rank == 0 && extra) {
        focus_count++;
    }
    int* local_b_ids = new int[focus_count];

    int start = rank * bin_per_proc + (rank == 0 ? 0 : (extra ? 1 : 0));
    for(int i = 0 ; i < focus_count ; i++ ) {
        int alternate_id = normal_to_alternate_sqmatrix_id(start + i, bin_row_count);
        local_b_ids[i] = alternate_id;
    }

    return local_b_ids;
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
bool inline has_topleft_bin(int bin_id) {
    return has_up_bin(bin_id) && has_left_bin(bin_id);
}
bool inline has_topright_bin(int bin_id) {
    return has_up_bin(bin_id) && has_right_bin(bin_id);
}
bool inline has_bottomleft_bin(int bin_id) {
    return has_down_bin(bin_id) && has_left_bin(bin_id);
}
bool inline has_bottomright_bin(int bin_id) {
    return has_down_bin(bin_id) && has_right_bin(bin_id);
}

vector<int>& get_bin_neightbors_id(int focus_bin_id) {
    vector<int>* neighbours = new vector<int>();
    if(has_topleft_bin(focus_bin_id)) {
        neighbours->push_back(focus_bin_id - bin_row_count - 1);
    }
    if(has_up_bin(focus_bin_id)) {
        neighbours->push_back(focus_bin_id - bin_row_count);
    }
    if(has_topright_bin(focus_bin_id)) {
        neighbours->push_back(focus_bin_id - bin_row_count + 1);
    }
    if(has_left_bin(focus_bin_id)) {
        neighbours->push_back(focus_bin_id - 1);
    }
    if(has_right_bin(focus_bin_id)) {
        neighbours->push_back(focus_bin_id + 1);
    }
    if(has_bottomleft_bin(focus_bin_id)) {
        neighbours->push_back(focus_bin_id + bin_row_count - 1);
    }
    if(has_down_bin(focus_bin_id)) {
        neighbours->push_back(focus_bin_id + bin_row_count);
    }
    if(has_bottomright_bin(focus_bin_id)) {
        neighbours->push_back(focus_bin_id + bin_row_count + 1);
    }

    return *neighbours;
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    printf("RANK %d:\n", rank);


    bin_row_count = floor(size / cutoff);
    bin_size = size / bin_row_count;

    printf("SIZE %f - BIN SIZE %f - CUTOFF %f\n", size, bin_size, cutoff);
    bin_count = bin_row_count * bin_row_count;

    if(bin_count > num_procs) {
        bin_per_proc = (int)(bin_count / num_procs);
    }
    else {
        bin_per_proc = 1;
    }
    focus_ids = get_focus_ids(rank);

    neighbour_ids = new vector<int>[focus_count];

    for(int i = 0; i < focus_count; ++i) {
        int focus_id = focus_ids[i];
        neighbour_ids[i] = get_bin_neightbors_id(focus_id);
    }

    bins.focuses = new focus[focus_count];
    for(int i = 0; i < focus_count; ++i) {
        focus* focus = &bins.focuses[i];

        bin_t* focus_bin = get_bin(focus_ids[i]);
        int neighbours_size = (int)neighbour_ids[i].size();
        focus->init(focus_bin, neighbours_size);

//        printf("F%d: ", focus->id());
        for(int j = 0; j < neighbours_size; ++j) {
            bin_t* neighbour_bin = get_bin(neighbour_ids[i].at(j));
            focus->neighbours[j] = neighbour_bin;
//            printf("%d ", neighbour_bin->id);
        }


//        printf("\n");
    }

    for(int i = 0; i < num_parts; ++i) {
        particle_t particle = parts[i];
        int particle_bin_id = get_bin_id(particle);
        bin_t* particle_bin = get_bin(particle_bin_id, false);
        if(particle_bin != nullptr) {
            particle_bin->add_particle(&particle);
            printf("Part%d(%f,%f) in bin %d\n", i, particle.x, particle.y, particle_bin_id);
        }
        else {
            printf("Part%d outside\n", i);
        }
    }

//    for (auto const& x : bin_data) {
//        printf("BIND ID: %d\n", x.first);
//        for(particle_t* particle : x.second->particles){
//            printf("%d:(%f,%f) -- ", x.first, particle->x, particle->y);
//        }
//    }


}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

//    for(int i = 0; i < focus_count; i++) {
//        neighbour_ids[i]->clear();
//        delete neighbour_ids[i];
//    }
//    delete focus_ids;
}