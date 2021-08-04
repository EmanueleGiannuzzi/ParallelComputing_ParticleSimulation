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

    unsigned long get_buffer_size() const{
        return (particles.size() * sizeof(particle_t)) + (2 * sizeof(int));
    }
};

//Remember to delete the buffer, please
char* serialize_bin_data(const bin_t* bin) {
    unsigned long  buffer_size = bin->get_buffer_size();
    char* buffer = new char[buffer_size];//TODO: Delete
    char* pointer = buffer;
    memcpy(pointer, &bin->id,  sizeof(int));
    pointer += sizeof(int);
    size_t size = bin->particles.size();
    memcpy(pointer, &size,  sizeof(size_t));
    pointer += sizeof(int);
    for(int i = 0; i < size; ++i) {
        memcpy(pointer, bin->particles[i],  sizeof(particle_t));
        pointer += sizeof(particle_t);
    }
    return buffer;
}

void deserialize_bin_data(const char buffer[], particle_t* parts, int parts_size) {
    const char* pointer = buffer;
    size_t id;
    memcpy(&id, pointer, sizeof(size_t));
    pointer += sizeof(int);
    int size = 0;
    memcpy(&size, pointer, sizeof(int));
    pointer += sizeof(int);
    for(int i = 0; i < size; ++i) {
        particle_t particle;
        memcpy(&particle, pointer, sizeof(particle_t));
        pointer += sizeof(particle_t);
        for(int j = 0; j < parts_size; ++j) {//TODO: Maybe use a map?
            if(parts[j].id == particle.id) {
                memcpy(&parts[j], &particle, sizeof(particle_t));
                break;
            }
        }
    }
}

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

int num_procs;
int my_rank;
double bin_size;
int bin_count;
int bin_row_count;
int bin_per_proc;
int focus_count;

int* focus_ids;
vector<int>* neighbour_ids;


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

// Integrate the ODE
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

    void apply_forces() {
        for(int i = 0; i < neighbours_size; ++i) {
            bin_t* neighbour_bin = neighbours[i];
            for(particle_t* focus_particle : focus_bin->particles){
                for(particle_t* neighbour_particle : neighbour_bin->particles){
                    apply_force(*focus_particle, *neighbour_particle);
                }
            }
        }
    }

    void move_particles(double size) {
        for(particle_t* focus_particle : focus_bin->particles){
            move(*focus_particle, size);
        }
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
        int alternate_id = normal_to_alternate_sqmatrix_id(start + i, bin_row_count);
        local_b_ids[i] = alternate_id;
    }

    return local_b_ids;
}

int get_rank_from_bin_id(int bin_id) {
    bin_id = normal_to_alternate_sqmatrix_id(bin_id, bin_row_count);

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

vector<int>& get_bin_neightbors_id(int focus_bin_id) {
    vector<int>* neighbours = new vector<int>();
    for(int i = 0; i<directions.size(); ++i) {
        if(directions_check[i](focus_bin_id)){
            neighbours->push_back(directions[i](focus_bin_id));
        }
    }
    return *neighbours;
}



MPI_Datatype MPI_BIN_TYPE;

void init_MPI_bin_type() {
    const int nitems = 7;
    int blocklengths[7] = {1, 1, 1, 1, 1, 1, 1};
    MPI_Datatype types[7] = {MPI_UINT64_T, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE,
                             MPI_DOUBLE,   MPI_DOUBLE, MPI_DOUBLE};
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int _num_procs) {
    printf("RANK %d:\n", rank);

    init_MPI_bin_type();

    num_procs = _num_procs;
    my_rank = rank;

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

        printf("F%d: ", focus->id());
        for(int j = 0; j < neighbours_size; ++j) {
            bin_t* neighbour_bin = get_bin(neighbour_ids[i].at(j));
            focus->neighbours[j] = neighbour_bin;
            printf("%d ", neighbour_bin->id);
        }


       printf("\n");
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

void send_to_neighbours(const focus* focuses, vector<MPI_Request>& requests) {
    for(int i = 0; i < focus_count; ++i)  {//SEND
        const focus* focus = &focuses[i];
        char* buffer = serialize_bin_data(focus->focus_bin);
        int buffer_size = focus->focus_bin->get_buffer_size();
        for(int dir_id = 0; dir_id < directions.size(); ++dir_id) {
            if(directions_check[dir_id](focus->focus_bin->id)) {
                int neighbour_bin_id = directions[dir_id](focus->id());
                int dest_rank = get_rank_from_bin_id(neighbour_bin_id);
                //TODO: If rank == my rank don't send
                if(dest_rank != my_rank) {
                    MPI_Request request;
                    MPI_Isend(buffer, buffer_size, MPI_BYTE, dest_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &request);
                    requests.push_back(request);
                }
            }
        }
    }
}

void receive_from_neighbours(const focus* focuses, particle_t* parts, int parts_size) {
    for(int i = 0; i < focus_count; ++i) {//SEND
        const focus *focus = &focuses[i];
        for(int dir_id = 0; dir_id < directions.size(); ++dir_id) {
            if(directions_check[dir_id](focus->focus_bin->id)) {
                int neighbour_bin_id = directions[dir_id](focus->id());
                int source_rank = get_rank_from_bin_id(neighbour_bin_id);
                MPI_Status status;
                MPI_Probe(source_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                int buffer_size;
                MPI_Get_count(&status, MPI_BYTE, &buffer_size);
                char buffer[buffer_size];
                MPI_Recv(buffer, buffer_size, MPI_BYTE, source_rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                deserialize_bin_data(buffer, parts, parts_size);
            }
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    focus* focuses = bins.focuses;
    for(int i = 0; i < focus_count; ++i)  {
        focuses[i].apply_forces();
    }
    for(int i = 0; i < focus_count; ++i)  {
        focuses[i].move_particles(size);
    }

    //TODO: Comunicazione
    vector<MPI_Request> requests;
    send_to_neighbours(focuses, requests);
    receive_from_neighbours(focuses, parts, num_parts);



    for(int i = 0; i < focus_count; ++i) {//RECEIVE
        focus *focus = &focuses[i];
    }
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