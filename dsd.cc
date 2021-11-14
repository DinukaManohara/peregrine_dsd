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

void densest_subgraph(int &&h, std::string &&graph) {
    DataGraph g(graph);

    std::cout << "Edge count : " << g.get_edge_count() << std::endl;
    std::cout << "Vertex count : " << g.get_vertex_count() << std::endl;
    std::cout << "--------------------" << std::endl;

    const std::uint32_t NUM_THREADS = 4;

    /*auto on_worker_sync = [](){ 
        std::cout << "Worker synced.";
    };*/
    
    std::uint32_t visited = 0;
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

    const std::uint32_t N = vertices.size();
    float max_core_density = static_cast<float>(clique_count) / static_cast<float>(N);

    // Worker threads to calculate the (k,h)-core values of the vertices.
    auto worker = [&visited, &N, &vertices, &degree_map, &cliques_map, &cliques, &sync_point](std::uint32_t for_start, std::uint32_t for_end) {  
        std::uint32_t l = 0; 
        std::uint32_t s = 0;
        std::uint32_t e = 0; 

        while (visited < N) {
            std::vector<std::uint32_t> buff;  
            
            for (std::uint32_t k = for_start; k < for_end; k++) {
                std::uint32_t v = vertices[k];
                if (degree_map[v] == l) {
                    buff.push_back(v);
                    e = e + 1;
                }
            }
            // Barrier sync
            sync_point.arrive_and_wait();

            while (s < e) {
                std::uint32_t v = buff[s];
                s = s + 1;

                for (std::uint32_t h : cliques_map[v]) {
                    for (std::uint32_t u : cliques[h]) {
                        std::uint32_t deg_u = degree_map[u];
                        if (deg_u > l) {
                            std::uint32_t du = __sync_fetch_and_sub(&degree_map[u], 1);
                            
                            if (du == (l + 1)) {
                                buff.push_back(u);
                                e = e + 1;
                            }

                            if( du <= l ) {
                                __sync_fetch_and_add(&degree_map[u], 1);
                            }
                        }
                    }
                }
            }
            
            __sync_fetch_and_add(&visited, e);

            // Barrier sync
            sync_point.arrive_and_wait();

            s = 0;
            e = 0;
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

        worker_threads.emplace_back(worker, for_start, for_end);
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
}

int main() { //(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    densest_subgraph(5, "data/citeseer/");
  
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
    std::cout << duration.count() << " microseconds consumed." << std::endl;
    
    return 0;
}