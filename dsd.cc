#include <iostream>
#include <vector>
#include <iterator>
#include <unordered_map>
#include <tuple>
//#include <barrier>
#include <pthread.h>
#include <mutex>
#include <thread>
#include <sstream>
#include <stack>
#include <limits>
#include <stdlib.h>
#include <chrono>
#include "Peregrine.hh"

using namespace std;
  
struct Edge
{
    // To store current flow and capacity of edge
    double flow, capacity;
  
    // An edge u--->v has start vertex as u and end
    // vertex as v.
    uint32_t v;
  
    Edge(double flow, double capacity, uint32_t v)
    {
        this->flow = flow;
        this->capacity = capacity;
        this->v = v;
    }
};
  
// Represent a Vertex
struct Vertex
{
    int h;
    double e_flow;
    int dfs_visited = 0;
  
    Vertex(int h = 0, double e_flow = 0.0)
    {
        this->h = h;
        this->e_flow = e_flow;
    }
};
  
// To represent a flow network
class Network {
    double V;    // No. of vertices
    //vector<Vertex> ver;
    unordered_map<uint32_t, Vertex> ver;
    unordered_map<uint32_t, vector<Edge>> adjecency_map;
  
    // Function to push excess flow from u
    bool push(uint32_t u);
  
    // Function to relabel a vertex u
    void relabel(uint32_t u);
  
    // This function is called to initialize
    // preflow
    void preflow(uint32_t s);
  
    // Function to reverse edge
    void update_reverse_edge_flow(uint32_t u, uint32_t v, double flow);
  
public:
    Network(double V);  // Constructor
  
    // function to add an edge to graph
    void add_edge(uint32_t u, uint32_t v, double w);
  
    // returns maximum flow from s to t
    void get_min_cut(uint32_t s, uint32_t t, vector<uint32_t> &s_cut);
};
  
Network::Network(double V){
    this->V = V;
}
  
void Network::add_edge(uint32_t u, uint32_t v, double capacity)
{
    // flow is initialized with 0 for all edge
    if (adjecency_map.contains(u)) {
        adjecency_map[u].push_back(Edge(0.0, capacity, v));
    } else {
        adjecency_map.insert({u, {Edge(0.0, capacity, v)}});
    }

    if (!ver.contains(u)) {
        Vertex u_ver = Vertex(0, 0.0);
        ver.insert({u, u_ver});
    }

    if (!ver.contains(v)) {
        Vertex v_ver = Vertex(0, 0.0);
        ver.insert({v, v_ver});
    }
}
  
void Network::preflow(uint32_t s)
{
    // Making h of source Vertex equal to no. of vertices
    // Height of other vertices is 0.
    ver[s].h = ver.size();
  
    //
    for (uint32_t i = 0; i < adjecency_map[s].size(); i++)
    {
        // Flow is equal to capacity
        adjecency_map[s][i].flow = adjecency_map[s][i].capacity;

        // Initialize excess flow for adjacent v
        //ver[edge.v].e_flow += edge.flow;
        ver[adjecency_map[s][i].v].e_flow += adjecency_map[s][i].capacity;

        // Add an edge from v to s in residual graph with
        // capacity equal to 0

        if (adjecency_map.contains(adjecency_map[s][i].v)) {
            adjecency_map[adjecency_map[s][i].v].push_back(Edge(-adjecency_map[s][i].flow, 0.0, s));
        } else {
            adjecency_map.insert({adjecency_map[s][i].v, {Edge(-adjecency_map[s][i].flow, 0.0, s)}});
        }
    }
}
  
// returns index of overflowing Vertex
uint32_t overflow_vertex(unordered_map<uint32_t, Vertex>& ver, uint32_t s, uint32_t t)
{
    for (auto [key, value] : ver) {
        if (key != s && key != t) {
            if (ver[key].e_flow > 0) {
                return key;
            }
        }
    }
  
    // 0 if no overflowing Vertex
    return 0;
}
  
// Update reverse flow for flow added on ith Edge
void Network::update_reverse_edge_flow(uint32_t vert_u, uint32_t vert_v, double flow)
{
    uint32_t u = vert_v, v = vert_u;
  
    if (adjecency_map.contains(u)) {
        for (uint32_t i = 0; i < adjecency_map[u].size(); i++)
        {
            if (adjecency_map[u][i].v == v)
            {
                adjecency_map[u][i].flow -= flow;
                return;
            }
        }
    }
  
    // adding reverse Edge in residual graph

    Edge e = Edge(0, flow, v);

    if (adjecency_map.contains(u)) {
        adjecency_map[u].push_back(e);
    } else {
        adjecency_map.insert({u, {e}});
    }
}
  
// To push flow from overflowing vertex u
bool Network::push(uint32_t u)
{
    // Traverse through all edges to find an adjacent (of u)
    // to which flow can be pushed
    for (uint32_t i = 0; i < adjecency_map[u].size(); i++)
    {
        // if flow is equal to capacity then no push
        // is possible
        if (adjecency_map[u][i].flow == adjecency_map[u][i].capacity) {
            continue;
        }

        // Push is only possible if height of adjacent
        // is smaller than height of overflowing vertex
        if (ver[u].h > ver[adjecency_map[u][i].v].h)
        {
            // Flow to be pushed is equal to minimum of
            // remaining flow on edge and excess flow.
            double flow = min(adjecency_map[u][i].capacity - adjecency_map[u][i].flow, ver[u].e_flow);
            
            // Reduce excess flow for overflowing vertex
            ver[u].e_flow -= flow;

            // Increase excess flow for adjacent
            ver[adjecency_map[u][i].v].e_flow += flow;

            // Add residual flow (With capacity 0 and negative
            // flow)
            adjecency_map[u][i].flow += flow;

            update_reverse_edge_flow(u, adjecency_map[u][i].v, flow);

            return true;
        }
    }

    return false;
}
  
// function to relabel vertex u
void Network::relabel(uint32_t u)
{
    // Initialize minimum height of an adjacent
    uint32_t mh = numeric_limits<uint32_t>::max();
  
    // Find the adjacent with minimum height
    for (uint32_t i = 0; i < adjecency_map[u].size(); i++)
    {
        // if flow is equal to capacity then no
        // relabeling
        if (adjecency_map[u][i].flow == adjecency_map[u][i].capacity) {
            continue;
        }

        // Update minimum height
        if (ver[adjecency_map[u][i].v].h < mh)
        {
            mh = ver[adjecency_map[u][i].v].h;

            // updating height of u
            ver[u].h = mh + 1;
        }
    }
}
  
// main function for printing maximum flow of graph
void Network::get_min_cut(uint32_t s, uint32_t t, vector<uint32_t> &s_cut) {
    preflow(s);
  
    // loop untill none of the Vertex is in overflow
    uint32_t v = overflow_vertex(ver, s, t);
    while (v != 0) {
        //cout << v << ", " << t << endl;
        if (!push(v)) {
            relabel(v);
        }
        v = overflow_vertex(ver, s, t);
    }
  
    // ver.back() returns last Vertex, whose
    // e_flow will be final maximum flow

    //cout << "Maximum flow is " << ver.back().e_flow << endl;

    //vector<uint32_t> s_cut;
    stack<uint32_t> stack;
    uint32_t u;

    stack.push(s);
    s_cut.push_back(s);
    ver[s].dfs_visited = 1;

    while ( !stack.empty() ) {
        u = stack.top();
        stack.pop();

        for (int i = 0; i < adjecency_map[u].size(); i++) {
            if (adjecency_map[u][i].flow < adjecency_map[u][i].capacity && ver[adjecency_map[u][i].v].dfs_visited == 0) {
                s_cut.push_back(adjecency_map[u][i].v);
                stack.push(adjecency_map[u][i].v);
                ver[adjecency_map[u][i].v].dfs_visited = 1;
            }
        }
    }

    /*cout << "S cut is :" << endl;

    for (int i = 0; i < s_cut.size(); i++) {
        cout << s_cut[i] << ", ";
    }
    cout << endl;*/
}


// DSD Portion
using namespace Peregrine;

void count_clique(int size, DataGraph &g, std::vector<std::vector<std::uint32_t>> &cliques) {
    int nthreads = 4;

    std::mutex write_mutex;
    
    std::vector<SmallGraph> patterns;
    SmallGraph clique = PatternGenerator::clique(size);
    patterns.push_back(clique);

    const auto callback = [&cliques, &write_mutex](auto &&handle, auto &&match)
    {
        //handle.map(match.pattern, 1);
        //handle.template output<CSV>(match.mapping); // this writes out matches
        //std::cout << match.mapping << std::endl;
        //std::cout << decltype() << std::endl;
        write_mutex.lock();
        cliques.push_back(match.mapping);
        write_mutex.unlock();
    };
    auto results = match<Pattern, uint64_t, AT_THE_END, UNSTOPPABLE>(g, patterns, nthreads, callback);
    //auto results = count(g, patterns, nthreads);
    /*int clique_count = 0;
    for (const auto &[pattern, count] : results) {
        clique_count = count;
    }

    return clique_count;*/
}

uint32_t core_decomposition(int &h, 
                        DataGraph &g, 
                        std::unordered_map<std::uint32_t, std::uint32_t> &densest_core_vertices,
                        std::vector<std::pair<std::uint32_t,std::uint32_t>> &densest_core_edge_list)
{
    //SmallGraph sg(graph);

    std::cout << "Edge count : " << g.get_edge_count() << std::endl;
    std::cout << "Vertex count : " << g.get_vertex_count() << std::endl;
    std::cout << "--------------------" << std::endl;

    const std::uint32_t TOTAL_VERTICES = g.get_vertex_count();
    const std::uint32_t NUM_THREADS = 4;

    /*auto on_worker_sync = [](){ 
        std::cout << "Worker synced.";
    };*/
    
    //std::barrier sync_point(NUM_THREADS);
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    std::vector<std::jthread> counter_threads;
    std::vector<std::jthread> worker_threads;
    std::vector<std::vector<std::uint32_t>> cliques;
    std::vector<std::uint32_t> vertices; 
    std::unordered_map<std::uint32_t, std::uint32_t> degree_map;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> cliques_map;

    count_clique(h, g, cliques);
    int clique_count = cliques.size();
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
    // This one could possibly be improved further for performance
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

    std::vector<uint32_t> clique_tracker(clique_count, 0);
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
            //sync_point.arrive_and_wait();
            pthread_barrier_wait (&barrier);

            while (s < e) {
                std::uint32_t v = buff[s];
                s = s + 1;
                
                for (std::uint32_t h : cliques_map[v]) {

                    uint32_t clique_visitation_count = __sync_fetch_and_add(&clique_tracker[h], 1);

                    if (clique_visitation_count > 0) {
                        continue;
                    } else if (clique_visitation_count == 0) {
                        thread_deleted_cliques += 1;
                    }

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
                        }
                    }
                }
            }
            
            __sync_fetch_and_add(&visited, e);
            __sync_fetch_and_add(&deleted_cliques, thread_deleted_cliques);

            // Barrier sync
            //sync_point.arrive_and_wait();
            pthread_barrier_wait (&barrier);

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

    std::unordered_map<std::uint32_t, std::uint32_t> densest_core_vertices_temp;
    
    std::uint32_t dc_ver_count = 0;
    std::uint32_t max_core_number = 0;

    for (const auto& [key, value] : degree_map) {
        if (value >= max_density_core_number) {
            dc_ver_count += 1;
            densest_core_vertices.insert({dc_ver_count, key});
            densest_core_vertices_temp.insert({key, dc_ver_count});
        }

        if (value > max_core_number) {
            max_core_number = value;
        }
    }

    for (auto& it : densest_core_vertices_temp) {
        for (std::uint32_t i = 0; i < g.get_adj(it.first).length; i++) {
            if (densest_core_vertices_temp.contains(g.get_adj(it.first).ptr[i])) {
                if (it.first < g.get_adj(it.first).ptr[i]) {
                    densest_core_edge_list.emplace_back(it.second, densest_core_vertices_temp[g.get_adj(it.first).ptr[i]]);
                }
            }
        }
    }

    for (auto& it : densest_core_edge_list) {
        std::cout << it.first << " -- " << it.second << ";" << std::endl;
    }

    SmallGraph densest_core_sg(densest_core_edge_list);
    DataGraph densest_core_dg(densest_core_sg);

    std::cout << "Edge count = " << densest_core_dg.get_edge_count() << std::endl;
    std::cout << "Vertex count = " << densest_core_dg.get_vertex_count() << std::endl;
    
    std::vector<SmallGraph> patterns_t;
    SmallGraph clique_t = PatternGenerator::clique(h);
    patterns_t.push_back(clique_t);
    
    auto results_t = count(densest_core_dg, patterns_t, 4);
    int clique_count_t = 0;
    for (const auto &[pattern_t, count] : results_t) {
        clique_count_t = count;
    }

    float dc_density = static_cast<float>(clique_count_t) / static_cast<float>(dc_ver_count);

    std::cout << "Clique count = " << clique_count_t << std::endl;
    std::cout << "Vertex count = " << dc_ver_count << std::endl;

    std::cout << "Densest core density calculated separately = " << dc_density << std::endl;

    return max_core_number;
}

void connected_components(std::vector<uint32_t> &vertices, 
                            std::vector<std::pair<std::uint32_t,std::uint32_t>> &edge_list,
                            std::vector<uint32_t> &p_vector) 
{
    const std::uint32_t NUM_THREADS = 4;

    std::vector<std::jthread> init_threads;

    /*std::vector<std::pair<std::uint32_t,std::uint32_t>> edge_list;

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

    SmallGraph g(sg);

    std::vector<uint32_t> vertices = g.v_list();*/

    uint32_t edge_count = edge_list.size();
    uint32_t vertex_count = vertices.size();

    std::cout << "Edge count : " << edge_count << std::endl;
    std::cout << "Vertex count : " << vertex_count << std::endl;

    //std::vector<uint32_t> p_vector(vertex_count+1);
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

void find_densest_subgraph(int &&h, std::string &&graph) {
    uint32_t NUM_THREADS = 4;
    std::vector<std::jthread> min_cut_threads;

    DataGraph g(graph);

    // Mapping of original vertices to the new ones
    std::unordered_map<std::uint32_t, std::uint32_t> densest_core_vertices;

    // Edge vector for the densest core
    std::vector<std::pair<std::uint32_t,std::uint32_t>> densest_core_edge_list;
    
    uint32_t max_core_number = core_decomposition(  h, 
                                                    g, 
                                                    densest_core_vertices, 
                                                    densest_core_edge_list);

    std::vector<uint32_t> vertices;

    for (uint32_t i = 1; i < densest_core_vertices.size() + 1; i++) {
        vertices.push_back(i);
    }

    // Parent vector to distinguish the connected componets
    std::vector<uint32_t> parent_vector(vertices.size()+1);

    connected_components(   vertices, 
                            densest_core_edge_list, 
                            parent_vector);

    std::unordered_map<uint32_t, std::vector<uint32_t>> components;
    std::vector<uint32_t> component_keys;

    for (uint32_t i = 1; i < parent_vector.size(); i++) {
        if (components.contains(parent_vector[i])) {
            components[parent_vector[i]].push_back(i);
        } else {
            components.insert({parent_vector[i], {i}});
        }
    }

    for (auto [key, value] : components) {
        component_keys.push_back(key);
    }

    SmallGraph dc_sg(densest_core_edge_list);
    DataGraph dc_dg(dc_sg);

    std::vector<std::vector<uint32_t>> h_cliques;
    std::vector<std::vector<uint32_t>> h_minus_one_cliques;
    
    count_clique(h, dc_dg, h_cliques);
    count_clique(h-1, dc_dg, h_minus_one_cliques);

    uint32_t h_clique_count = h_cliques.size();

    std::unordered_map<uint32_t, std::vector<uint32_t>> h_clique_map;

    for (uint32_t i = 0; i < h_clique_count; i++) {
        for (uint32_t u : h_cliques[i]) {
            if (h_clique_map.contains(u)) {
                h_clique_map[u].push_back(i);
            } else {
                h_clique_map.insert({u, {i}});
            }
        }
    }

    double dc_density = static_cast<double>(h_clique_count) / static_cast<double>(vertices.size());

    std::mutex write_mutex;
    double max_density = dc_density;
    std::unordered_map<uint32_t, int> ds_vertices;

    auto min_cut_worker = [&](uint32_t for_start, uint32_t for_end) {
        for (uint32_t k = for_start; k < for_end; k++) {
            std::vector<uint32_t> u_with_s = components[component_keys[k]];
            uint32_t s = 0; uint32_t t = h_minus_one_cliques.size() + vertices.size() + 1;
            
            double u = static_cast<double>(max_core_number);
            double l = dc_density;
            double alpha;

            double component_size = static_cast<double>(components[component_keys[k]].size());

            while ( u - l >= (1.0 / (component_size * (component_size - 1.0))) ) {
                
                alpha = (l + u) / 2.0;

                std::cout << alpha << std::endl;

                Network net(component_size);

                for (uint32_t v : components[component_keys[k]]) {
                    net.add_edge(s, v, h_clique_map[v].size());
                    net.add_edge(v, t, alpha * h);// alpha stuff goes here);
                }

                for (uint32_t i = 0; i < h_minus_one_cliques.size(); i++) {
                    if (parent_vector[h_minus_one_cliques[i][0]] == component_keys[k]) {
                        uint32_t fi = vertices.size() + i + 1;

                        for (uint32_t v : h_minus_one_cliques[i]) {
                            net.add_edge(fi, v, numeric_limits<double>::infinity());
                        }

                        uint32_t conns;
                        for (uint32_t w : components[component_keys[k]]) {
                            conns = 0;
                            for (uint32_t v : h_minus_one_cliques[i]) {
                                for (uint32_t j = 0; j < g.get_adj(densest_core_vertices[w]).length; j++) {
                                    if (g.get_adj(densest_core_vertices[w]).ptr[j] == densest_core_vertices[v]) {
                                        conns += 1;
                                        break;
                                    }
                                }
                            }

                            if (conns >= h - 1) {
                                net.add_edge(w, fi, 1);
                            }
                        }
                    }
                }

                std::vector<uint32_t> s_cut;

                net.get_min_cut(s, t, s_cut);

                if (s_cut.size() == 1 && s_cut[0] == s) {
                    u = alpha;
                } else {
                    l = alpha;
                    u_with_s = s_cut;
                }
            }

            std::unordered_map<uint32_t, int> u_map;
            for (auto v : u_with_s) {
                if (h_clique_map.contains(v)) {
                    u_map[v] = 1;
                }
            }

            std::unordered_map<uint32_t, int> vanishing_h_cliques;
            for (auto v : components[component_keys[k]]) {
                if ( !u_map.contains(v) ) {
                    for (auto i : h_clique_map[v]) {
                        if ( !vanishing_h_cliques.contains(i) ) {
                            vanishing_h_cliques[i] = 1;
                        }
                    }
                }
            }

            double new_density = 0;

            if (u_map.size() > 0) {
                new_density = static_cast<double>(h_clique_count - vanishing_h_cliques.size()) / static_cast<double>(u_map.size());
            }

            write_mutex.lock();

            if (new_density >= max_density) {
                max_density = new_density;
                ds_vertices = u_map;
            }

            write_mutex.unlock();
        }
    };

    // Distributing the components among the threads.
    uint32_t cc_count = component_keys.size();
    uint32_t lower = cc_count - (cc_count % NUM_THREADS);
    uint32_t upper = lower + NUM_THREADS;
    uint32_t lower_diff = cc_count - lower;
    uint32_t upper_diff = upper - cc_count;
    uint32_t chunk = 0;

    if (lower_diff < upper_diff) {
        chunk = lower / NUM_THREADS;
    } else {
        chunk = upper / NUM_THREADS;
    }

    std::cout << "Min-cut threads initialized." << std::endl;
    std::cout << "--------------------" << std::endl;
    
    if (cc_count >= NUM_THREADS) {
        for (std::uint32_t j = 0; j < NUM_THREADS; j++) {
            std::uint32_t for_start = j * chunk;
            std::uint32_t for_end = (j + 1) * chunk;

            if (j == NUM_THREADS - 1) {
                for_end = cc_count;
            }

            min_cut_threads.emplace_back(min_cut_worker, for_start, for_end);
        }
    } else {
        for (std::uint32_t j = 0; j < cc_count; j++) {
            std::uint32_t for_start = j;
            std::uint32_t for_end = (j + 1);

            min_cut_threads.emplace_back(min_cut_worker, for_start, for_end);
        }
    }

    for (auto& min_cut_thread : min_cut_threads) {
        min_cut_thread.join();
    }

    std::cout << "Min-cut threads finalized." << std::endl;
    std::cout << "--------------------" << std::endl;

    
    /*for (uint32_t k = 0; k < component_keys.size(); k++) {
        std::vector<uint32_t> u_with_s;
        uint32_t s = 0; uint32_t t = h_minus_one_cliques.size() + vertices.size() + 1;
        
        double u = static_cast<double>(max_core_number);
        double l = dc_density;
        double alpha;

        double component_size = static_cast<double>(components[component_keys[k]].size());

        while ( u - l >= (1.0 / (component_size * (component_size - 1.0))) ) {
            
            alpha = (l + u) / 2.0;

            std::cout << alpha << std::endl;

            Network net(component_size);

            for (uint32_t v : components[component_keys[k]]) {
                net.add_edge(s, v, h_clique_map[v].size());
                net.add_edge(v, t, alpha * h);// alpha stuff goes here);
            }

            for (uint32_t i = 0; i < h_minus_one_cliques.size(); i++) {
                if (parent_vector[h_minus_one_cliques[i][0]] == component_keys[k]) {
                    uint32_t fi = vertices.size() + i + 1;

                    for (uint32_t v : h_minus_one_cliques[i]) {
                        net.add_edge(fi, v, numeric_limits<double>::infinity());
                    }

                    uint32_t conns;
                    for (uint32_t w : components[component_keys[k]]) {
                        conns = 0;
                        for (uint32_t v : h_minus_one_cliques[i]) {
                            for (uint32_t j = 0; j < g.get_adj(densest_core_vertices[w]).length; j++) {
                                if (g.get_adj(densest_core_vertices[w]).ptr[j] == densest_core_vertices[v]) {
                                    conns += 1;
                                    break;
                                }
                            }
                        }

                        if (conns >= h - 1) {
                            net.add_edge(w, fi, 1);
                        }
                    }
                }
            }

            std::vector<uint32_t> s_cut;

            net.get_min_cut(s, t, s_cut);

            if (s_cut.size() == 1 && s_cut[0] == s) {
                u = alpha;
            } else {
                l = alpha;
                u_with_s = s_cut;
            }
        }

        std::unordered_map<uint32_t, int> u_map;
        for (auto v : u_with_s) {
            if (h_clique_map.contains(v)) {
                u_map[v] = 1;
            }
        }

        std::unordered_map<uint32_t, int> vanishing_h_cliques;
        for (auto v : components[component_keys[k]]) {
            if ( !u_map.contains(v) ) {
                for (auto i : h_clique_map[v]) {
                    if ( !vanishing_h_cliques.contains(i) ) {
                        vanishing_h_cliques[i] = 1;
                    }
                }
            }
        }

        double new_density = 0;

        if (u_map.size() > 0) {
            new_density = static_cast<double>(h_clique_count - vanishing_h_cliques.size()) / static_cast<double>(u_map.size());
        }

        if (new_density > max_density) {
            max_density = new_density;
            ds_vertices = u_map;
        }
    }*/

    std::cout << "Max density = " << max_density << std::endl;

    std::cout << "Densest subgraph (pseudo)" << std::endl;
    std::cout << "----------------" << std::endl;

    for (auto [key, value] : ds_vertices) {
        std::cout << key << ", ";
    }

    std::cout << std::endl;

    std::cout << "Densest subgraph (original)" << std::endl;
    std::cout << "----------------" << std::endl;

    for (auto [key, value] : ds_vertices) {
        std::cout << densest_core_vertices[key] << ", ";
    }

    std::cout << std::endl;
}

int main() { //(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    find_densest_subgraph(5, "data/netscience/");
  
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
    std::cout << duration.count() / 1000000.0 << " seconds consumed." << std::endl;
    
    return 0;
}