#include <iostream>
#include <vector>
#include <iterator>
#include <unordered_map>
#include <barrier>
#include <thread>
#include <sstream>
#include <stdlib.h>
#include <chrono>
#include "Peregrine.hh"

using namespace Peregrine;

int count_clique(int size, DataGraph &g, std::vector<std::vector<std::uint32_t>> &cliques) {
    int nthreads = 4;
    
    std::vector<SmallGraph> patterns;
    SmallGraph clique = PatternGenerator::clique(size);
    patterns.push_back(clique);

    const auto callback = [&cliques](auto &&handle, auto &&match)
    {
        handle.map(match.pattern, 1);
        //handle.template output<CSV>(match.mapping); // this writes out matches
        //std::cout << match.mapping << std::endl;
        //std::cout << decltype() << std::endl;

        cliques.push_back(match.mapping);
    };
    auto results = match<Pattern, uint64_t, AT_THE_END, UNSTOPPABLE>(g, patterns, nthreads, callback);
    //auto results = count(g, patterns, nthreads);
    int clique_count = 0;
    for (const auto &[pattern, count] : results) {
        clique_count = count;
    }

    return clique_count;
}

void core_decomposition(int &&h, std::string &&graph) {
    DataGraph g(graph);

    std::cout << "Edge count : " << g.get_edge_count() << std::endl;
    std::cout << "Vertex count : " << g.get_vertex_count() << std::endl;
    std::cout << "--------------------" << std::endl;

    const std::uint32_t TOTAL_VERTICES = g.get_vertex_count();
    const std::uint32_t NUM_THREADS = 4;

    /*auto on_worker_sync = [](){ 
        std::cout << "Worker synced.";
    };*/
    
    std::barrier sync_point(NUM_THREADS);//, on_worker_sync);
    std::vector<std::jthread> counter_threads;
    std::vector<std::jthread> worker_threads;
    std::vector<std::vector<std::uint32_t>> cliques;
    std::vector<std::uint32_t> vertices; 
    std::unordered_map<std::uint32_t, std::uint32_t> degree_map;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> cliques_map;

    int clique_count = count_clique(h, g, cliques);
    std::cout << "Count: " << clique_count << std::endl;
    std::cout << "--------------------" << std::endl;

    // Counter threads to count the clique-degree for each vertex present in the matched cliques.
    auto counter = [&cliques](std::uint32_t for_start, std::uint32_t for_end, std::unordered_map<std::uint32_t, std::uint32_t>* count_map, std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>* clique_map) {  
        for (std::uint32_t a = for_start; a < for_end; a++) {
            for (std::uint32_t b: cliques[a]) {
                if (count_map->contains(b)) {
                    count_map->at(b) += 1;
                } else {
                    count_map->insert({b, 1});
                }

                if (clique_map->contains(b)) {
                    clique_map->at(b).push_back(a);
                } else {
                    clique_map->insert({b ,{a}});
                }
            }
        }
    };

    std::vector<std::unordered_map<std::uint32_t, std::uint32_t>> count_map_holder(NUM_THREADS);
    std::vector<std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>> clique_map_holder(NUM_THREADS);

    // Distributing the found cliques among the counter threads.
    std::uint32_t lower = clique_count - (clique_count % NUM_THREADS);
    std::uint32_t upper = lower + NUM_THREADS;
    std::uint32_t lower_diff = clique_count - lower;
    std::uint32_t upper_diff = upper - clique_count;
    std::uint32_t chunk;

    if (lower_diff < upper_diff) {
        chunk = lower / NUM_THREADS;
    } else {
        chunk = upper / NUM_THREADS;
    }

    std::cout << "Counter threads initialized." << std::endl;
    std::cout << "--------------------" << std::endl;

    for (std::uint32_t j = 0; j < NUM_THREADS; j++) {
        std::uint32_t for_start = j * chunk;
        std::uint32_t for_end = (j + 1) * chunk;

        if (j == NUM_THREADS - 1) {
            for_end = clique_count;
        }
        
        counter_threads.emplace_back(counter, for_start, for_end, &count_map_holder[j], &clique_map_holder[j]);
    }

    for (auto& counter_thread : counter_threads) {
        counter_thread.join();
    }

    std::cout << "Counter threads finalized." << std::endl;
    std::cout << "--------------------" << std::endl;

    // Implement the algorithm to aggregate the count_maps and the clique_maps
    // #######################################################################
    for (std::uint32_t j = 0; j < NUM_THREADS; j++) {
        for (const auto& [key, value] : count_map_holder[j]) {
            if (degree_map.contains(key)) {
                degree_map[key] += value;
            } else {
                degree_map[key] = value;
                vertices.push_back(key);
            }
        }

        for (const auto& [key, value] : clique_map_holder[j]) {
            if (cliques_map.contains(key)) {
                cliques_map[key].insert(
                    cliques_map[key].end(), 
                    std::make_move_iterator(value.begin()), 
                    std::make_move_iterator(value.end())
                );
            } else {
                cliques_map[key] = value;
            }
        }
    }

    // Printing operations
    // #######################################################################

    std::cout << "Degree map:" << std::endl;
    for (const auto& [key, value] : degree_map) {
        std::cout << key << " : " << value << std::endl;
    }

    std::cout << "--------------------" << std::endl;

    std::cout << "Cliques map:" << std::endl;
    for (const auto& [key, value] : cliques_map) {
        std::cout << key << " : ";
        for (auto c : value) {
            std::cout << c << ", ";
        }
        std::cout << std::endl;
    }

    std::cout << "--------------------" << std::endl;

    std::cout << "Cliques:" << std::endl;
    for (auto clique : cliques) {
        for (auto c : clique) {
            std::cout << c << "-";
        }
        std::cout << std::endl;
    }

    std::cout << "--------------------" << std::endl;

    std::cout << "Vertices: " << vertices.size() << std::endl;
    for (auto v : vertices) {
        std::cout << v << ", ";
    }
    std::cout << std::endl;
    std::cout << "--------------------" << std::endl;

    // End of printing operations
    // #######################################################################

    std::uint32_t visited = 0;
    std::uint32_t deleted_cliques = 0;
    const std::uint32_t N = vertices.size();
    std::uint32_t max_density_core_number = 0;
    float max_core_density = 0.0f;

    // Calculating the initial maximum core density (0th core and the 1st core)
    // #######################################################################
    float zero_core_density = static_cast<float>(clique_count) / static_cast<float>(TOTAL_VERTICES);
    float first_core_density = static_cast<float>(clique_count) / static_cast<float>(N);

    if (first_core_density >= zero_core_density) {
        max_core_density = first_core_density;
    } else {
        max_core_density = zero_core_density;
    }
    // #######################################################################

    // Worker threads to calculate the (k,h)-core values of the vertices.
    auto worker = [&](std::uint32_t thread_id, std::uint32_t for_start, std::uint32_t for_end) {  
        std::uint32_t l = 1; 
        std::uint32_t s = 0;
        std::uint32_t e = 0; 
        std::uint32_t thread_deleted_cliques = 0;
        float new_core_density = 0.0f;
        bool is_any_deleted = false;

        while (visited < N) {
            std::vector<std::uint32_t> buff; 
            //std::unordered_map<std::uint32_t, std::uint32_t> buff_map; 

            for (std::uint32_t k = for_start; k < for_end; k++) {
                std::uint32_t v = vertices[k];
                if (degree_map[v] == l) {
                    buff.push_back(v);
                    //buff_map.insert(v, 0);
                    e = e + 1;
                }
            }
            // Barrier sync
            sync_point.arrive_and_wait();

            while (s < e) {
                std::uint32_t v = buff[s];
                s = s + 1;
                
                for (std::uint32_t h : cliques_map[v]) {
                    is_any_deleted = false;

                    for (std::uint32_t u : cliques[h]) {
                        std::uint32_t deg_u = degree_map[u];
                        if (deg_u > l) {
                            std::uint32_t du = __sync_fetch_and_sub(&degree_map[u], 1);
                            
                            if (du == (l + 1)) {
                                buff.push_back(u);
                                e = e + 1;
                            }

                            if ( du <= l ) {
                                __sync_fetch_and_add(&degree_map[u], 1);
                            }

                            if (du > l) {
                                is_any_deleted = true;
                            }
                        }
                    }

                    if (is_any_deleted) {
                        thread_deleted_cliques += 1;
                    }
                }
            }
            
            __sync_fetch_and_add(&visited, e);
            __sync_fetch_and_add(&deleted_cliques, thread_deleted_cliques);

            // Barrier sync
            sync_point.arrive_and_wait();

            if (thread_id == 0) {
                if (N > visited) {
                    new_core_density = static_cast<float>(clique_count - deleted_cliques) / static_cast<float>(N - visited);

                    if (new_core_density >= max_core_density) {
                        max_core_density = new_core_density;
                        max_density_core_number = l + 1;
                    }
                }
            }

            s = 0;
            e = 0;
            thread_deleted_cliques = 0;
            l = l + 1;
        }
    };

    // Distributing the vertices among the worker threads.
    lower = vertices.size() - (vertices.size() % NUM_THREADS);
    upper = lower + NUM_THREADS;
    lower_diff = vertices.size() - lower;
    upper_diff = upper - vertices.size();

    if (lower_diff < upper_diff) {
        chunk = lower / NUM_THREADS;
    } else {
        chunk = upper / NUM_THREADS;
    }

    std::cout << "Worker threads initialized." << std::endl;
    std::cout << "--------------------" << std::endl;
    
    for (std::uint32_t j = 0; j < NUM_THREADS; j++) {
        std::uint32_t for_start = j * chunk;
        std::uint32_t for_end = (j + 1) * chunk;

        if (j == NUM_THREADS - 1) {
            for_end = vertices.size();
        }

        worker_threads.emplace_back(worker, j, for_start, for_end);
    }

    for (auto& worker_thread : worker_threads) {
        worker_thread.join();
    }

    std::cout << "Worker threads finalized." << std::endl;
    std::cout << "--------------------" << std::endl;

    std::cout << "(k,h)-core values:" << std::endl;
    for (const auto& [key, value] : degree_map) {
        std::cout << key << " : " << value << std::endl;
    }
    std::cout << "--------------------" << std::endl;

    std::cout << "Max density core number: " << max_density_core_number << std::endl;
    std::cout << "Max core density: " << max_core_density << std::endl;
    std::cout << "Deleted cliques: " << deleted_cliques << std::endl;
    std::cout << "--------------------" << std::endl;
}

void connected_components(std::string &&inputfile) {
    const std::uint32_t NUM_THREADS = 4;

    std::vector<std::jthread> init_threads;

    std::vector<std::pair<std::uint32_t,std::uint32_t>> edge_list;

    std::ifstream query_graph(inputfile.c_str());
    std::string line;
    while (std::getline(query_graph, line))
    {
        std::istringstream iss(line);
        std::vector<uint32_t> vs(std::istream_iterator<uint32_t>{iss}, std::istream_iterator<uint32_t>());

        uint32_t a, b;
        if (vs.size() == 2) {
            a = vs[0]; b = vs[1];
            edge_list.emplace_back(a, b);
        }
    }

    SmallGraph g(edge_list);

    std::vector<uint32_t> vertices = g.v_list();

    uint32_t edge_count = g.num_true_edges();
    uint32_t vertex_count = g.num_vertices();

    std::cout << "Edge count : " << edge_count << std::endl;
    std::cout << "Vertex count : " << vertex_count << std::endl;

    std::vector<uint32_t> p_vector(vertex_count+1);
    std::vector<uint32_t> gp_vector(vertex_count+1);
    bool change = true;

    auto initializer = [&](uint32_t for_start, uint32_t for_end) {
        for (uint32_t k = for_start; k < for_end; k++) {
            p_vector[vertices[k]] = vertices[k];
            gp_vector[vertices[k]] = vertices[k];
        }
    };

    auto hooking_worker = [&](uint32_t for_start, uint32_t for_end) {
        for (uint32_t k = for_start; k < for_end; k++) {
            uint32_t u = edge_list[k].first;
            uint32_t v = edge_list[k].second;
            uint32_t gp_u = gp_vector[u];
            uint32_t gp_v = gp_vector[v];
            
            if (gp_u > gp_v) {
                uint32_t p_u = p_vector[u];
                p_vector[p_u] = gp_v;
                p_vector[u] = gp_v;
            }

            if (gp_v > gp_u) {
                uint32_t p_v = p_vector[v];
                p_vector[p_v] = gp_u;
                p_vector[v] = gp_u;
            }
        }
    };

    auto shortcutting_worker = [&](uint32_t for_start, uint32_t for_end) {
        for (uint32_t k = for_start; k < for_end; k++) {
            uint32_t p_u = p_vector[vertices[k]];
            uint32_t gp_u = gp_vector[vertices[k]];
            
            if (p_u > gp_u) {
                p_vector[vertices[k]] = gp_u;
            }
        }
    };

    auto gp_calculator = [&](uint32_t for_start, uint32_t for_end) {
        for (uint32_t k = for_start; k < for_end; k++) {
            uint32_t p_u = p_vector[vertices[k]];
            uint32_t pp_u = p_vector[p_u];
            uint32_t gp_u = gp_vector[vertices[k]];
            
            if (pp_u != gp_u) {
                change = true;
            }

            gp_vector[vertices[k]] = pp_u;
        }
    };

    // Distributing the vertices among the threads.
    uint32_t v_lower = vertex_count - (vertex_count % NUM_THREADS);
    uint32_t v_upper = v_lower + NUM_THREADS;
    uint32_t v_lower_diff = vertex_count - v_lower;
    uint32_t v_upper_diff = v_upper - vertex_count;
    uint32_t v_chunk = 0;

    if (v_lower_diff < v_upper_diff) {
        v_chunk = v_lower / NUM_THREADS;
    } else {
        v_chunk = v_upper / NUM_THREADS;
    }

    // Distributing the edges among the threads.
    uint32_t e_lower = edge_count - (edge_count % NUM_THREADS);
    uint32_t e_upper = e_lower + NUM_THREADS;
    uint32_t e_lower_diff = edge_count - e_lower;
    uint32_t e_upper_diff = e_upper - edge_count;
    uint32_t e_chunk = 0;

    if (e_lower_diff < e_upper_diff) {
        e_chunk = e_lower / NUM_THREADS;
    } else {
        e_chunk = e_upper / NUM_THREADS;
    }

    std::cout << "Init threads initialized." << std::endl;
    std::cout << "--------------------" << std::endl;
    
    for (std::uint32_t j = 0; j < NUM_THREADS; j++) {
        std::uint32_t for_start = j * v_chunk;
        std::uint32_t for_end = (j + 1) * v_chunk;

        if (j == NUM_THREADS - 1) {
            for_end = vertex_count;
        }

        init_threads.emplace_back(initializer, for_start, for_end);
    }

    for (auto& init_thread : init_threads) {
        init_thread.join();
    }

    std::cout << "Init threads finalized." << std::endl;
    std::cout << "--------------------" << std::endl;

    int num_iter = 0;

    while (change) {
        std::vector<std::jthread> hooking_threads;
        std::vector<std::jthread> shortcut_threads;
        std::vector<std::jthread> gp_calc_threads;
        
        change = false;
        num_iter++;

        //std::cout << "Hooking threads initialized." << std::endl;
        //std::cout << "--------------------" << std::endl;
        
        for (std::uint32_t j = 0; j < NUM_THREADS; j++) {
            std::uint32_t for_start = j * e_chunk;
            std::uint32_t for_end = (j + 1) * e_chunk;

            if (j == NUM_THREADS - 1) {
                for_end = edge_count;
            }

            hooking_threads.emplace_back(hooking_worker, for_start, for_end);
        }

        for (auto& hooking_thread : hooking_threads) {
            hooking_thread.join();
        }

        //std::cout << "Hooking threads finalized." << std::endl;
        //std::cout << "--------------------" << std::endl;

        //std::cout << "Shortcutting threads initialized." << std::endl;
        //std::cout << "--------------------" << std::endl;
        
        for (std::uint32_t j = 0; j < NUM_THREADS; j++) {
            std::uint32_t for_start = j * v_chunk;
            std::uint32_t for_end = (j + 1) * v_chunk;

            if (j == NUM_THREADS - 1) {
                for_end = vertex_count;
            }

            shortcut_threads.emplace_back(shortcutting_worker, for_start, for_end);
        }

        for (auto& shortcut_thread : shortcut_threads) {
            shortcut_thread.join();
        }

        //std::cout << "Shortcutting threads finalized." << std::endl;
        //std::cout << "--------------------" << std::endl;

        //std::cout << "GP calculating threads initialized." << std::endl;
        //std::cout << "--------------------" << std::endl;
        
        for (std::uint32_t j = 0; j < NUM_THREADS; j++) {
            std::uint32_t for_start = j * v_chunk;
            std::uint32_t for_end = (j + 1) * v_chunk;

            if (j == NUM_THREADS - 1) {
                for_end = vertex_count;
            }

            gp_calc_threads.emplace_back(gp_calculator, for_start, for_end);
        }

        for (auto& gp_calc_thread : gp_calc_threads) {
            gp_calc_thread.join();
        }

        //std::cout << "GP calculating threads finalized." << std::endl;
        //std::cout << "--------------------" << std::endl;
    }

    std::cout << "FastSV took " << num_iter << " iterations" << std::endl;

    for (auto u : p_vector) {
        std::cout << u << " , ";
    }
    std::cout << std::endl;
}

int main() { //(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    //densest_subgraph(5, "data/citeseer/");

    connected_components("data/test_graph.txt");
  
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
    std::cout << duration.count() / 1000000.0 << " seconds consumed." << std::endl;
    
    return 0;
}