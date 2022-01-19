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

void find_densest_subgraph(int &&h, std::string &&graph) {
    uint32_t NUM_THREADS = 4;
    std::vector<std::jthread> min_cut_threads;

    DataGraph g(graph);

    std::vector<std::vector<uint32_t>> h_cliques;
    std::vector<std::vector<uint32_t>> h_minus_one_cliques;
    
    count_clique(h, g, h_cliques);
    count_clique(h-1, g, h_minus_one_cliques);

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

    uint32_t max_clique_degree = 0;
    for (auto [key, value] : h_clique_map) {
        if (value.size() > max_clique_degree) {
            max_clique_degree = value.size();
        }
    }

    double dc_density = static_cast<double>(h_clique_count) / static_cast<double>(g.get_vertex_count());

    double max_density = dc_density;
    std::unordered_map<uint32_t, int> ds_vertices;
    
    std::vector<uint32_t> u_with_s;
    uint32_t s = 0; uint32_t t = g.get_vertex_count() + 1;
    
    double u = static_cast<double>(max_clique_degree);
    double l = 0;
    double alpha;

    double n = static_cast<double>(g.get_vertex_count());

    while ( u - l >= (1.0 / (n * (n - 1.0))) ) {
        
        alpha = (l + u) / 2.0;

        std::cout << l << ", " << alpha << ", " << u << std::endl;

        Network net(n);

        for (uint32_t v = 1; v < g.get_vertex_count() + 1; v++) {
            net.add_edge(s, v, h_clique_map[v].size());
            net.add_edge(v, t, alpha * h);// alpha stuff goes here);
        }

        for (uint32_t i = 0; i < h_minus_one_cliques.size(); i++) {
            uint32_t fi = g.get_vertex_count() + i + 1;

            for (uint32_t v : h_minus_one_cliques[i]) {
                net.add_edge(fi, v, numeric_limits<double>::infinity());
            }

            uint32_t conns;
            for (uint32_t w = 1; w < g.get_vertex_count() + 1; w++) {
                conns = 0;
                for (uint32_t v : h_minus_one_cliques[i]) {
                    for (uint32_t j = 0; j < g.get_adj(w).length; j++) {
                        if (g.get_adj(w).ptr[j] == v) {
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

        std::vector<uint32_t> s_cut;

        net.get_min_cut(s, t, s_cut);

        if (s_cut.size() == 1 && s_cut[0] == s) {
            u = alpha;
        } else {
            l = alpha;
            u_with_s = s_cut;
        }
    }

    for (auto v : u_with_s) {
        if (h_clique_map.contains(v)) {
            ds_vertices[v] = 1;
        }
    }

    std::unordered_map<uint32_t, int> vanishing_h_cliques;
    for (uint32_t v = 1; v < g.get_vertex_count() + 1; v++) {
        if ( !ds_vertices.contains(v) ) {
            for (auto i : h_clique_map[v]) {
                if ( !vanishing_h_cliques.contains(i) ) {
                    vanishing_h_cliques[i] = 1;
                }
            }
        }
    }

    double new_density = 0;

    if (ds_vertices.size() > 0) {
        new_density = static_cast<double>(h_clique_count - vanishing_h_cliques.size()) / static_cast<double>(ds_vertices.size());
    }

    if (new_density >= max_density) {
        max_density = new_density;
    }

    std::cout << "Max density = " << max_density << std::endl;

    std::cout << "Densest subgraph" << std::endl;
    std::cout << "----------------" << std::endl;

    for (auto [key, value] : ds_vertices) {
        std::cout << key << ", ";
    }

    std::cout << std::endl;
}

int main() { //(int argc, char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    find_densest_subgraph(4, "data/netscience/");
  
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  
    std::cout << duration.count() / 1000000.0 << " seconds consumed." << std::endl;
    
    return 0;
}