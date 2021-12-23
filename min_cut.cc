#include <vector>
#include <unordered_map>
#include <iostream>
#include <stack>
#include <limits>
#include <stdlib.h>

using namespace std;
  
struct Edge
{
    // To store current flow and capacity of edge
    int flow, capacity;
  
    // An edge u--->v has start vertex as u and end
    // vertex as v.
    uint32_t v;
  
    Edge(int flow, int capacity, uint32_t v)
    {
        this->flow = flow;
        this->capacity = capacity;
        this->v = v;
    }
};
  
// Represent a Vertex
struct Vertex
{
    int h, e_flow;
    int dfs_visited = 0;
  
    Vertex(int h = 0, int e_flow = 0)
    {
        this->h = h;
        this->e_flow = e_flow;
    }
};
  
// To represent a flow network
class Network
{
    uint32_t V;    // No. of vertices
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
    void update_reverse_edge_flow(uint32_t u, uint32_t v, int flow);
  
public:
    Network(uint32_t V);  // Constructor
  
    // function to add an edge to graph
    void add_edge(uint32_t u, uint32_t v, int w);
  
    // returns maximum flow from s to t
    void get_min_cut(uint32_t s, uint32_t t);
};
  
Network::Network(uint32_t V)
{
    this->V = V;

    // all vertices are initialized with 0 height
    // and 0 excess flow
    /*for (uint32_t i = 0; i < V; i++) {
        ver.push_back(Vertex(0, 0));
    }*/
}
  
void Network::add_edge(uint32_t u, uint32_t v, int capacity)
{
    // flow is initialized with 0 for all edge
    if (adjecency_map.contains(u)) {
        adjecency_map[u].push_back(Edge(0, capacity, v));
    } else {
        adjecency_map.insert({u, {Edge(0, capacity, v)}});
    }

    if (!ver.contains(u)) {
        Vertex u_ver = Vertex(0, 0);
        ver.insert({u, u_ver});
    }

    if (!ver.contains(v)) {
        Vertex v_ver = Vertex(0, 0);
        ver.insert({v, v_ver});
    }
}
  
void Network::preflow(uint32_t s)
{
    // Making h of source Vertex equal to no. of vertices
    // Height of other vertices is 0.
    ver[s].h = ver.size();
  
    //
    for (int i = 0; i < adjecency_map[s].size(); i++)
    {
        // Flow is equal to capacity
        adjecency_map[s][i].flow = adjecency_map[s][i].capacity;

        // Initialize excess flow for adjacent v
        //ver[edge.v].e_flow += edge.flow;
        ver[adjecency_map[s][i].v].e_flow += adjecency_map[s][i].capacity;

        // Add an edge from v to s in residual graph with
        // capacity equal to 0

        if (adjecency_map.contains(adjecency_map[s][i].v)) {
            adjecency_map[adjecency_map[s][i].v].push_back(Edge(-adjecency_map[s][i].flow, 0, s));
        } else {
            adjecency_map.insert({adjecency_map[s][i].v, {Edge(-adjecency_map[s][i].flow, 0, s)}});
        }
    }
}
  
// returns index of overflowing Vertex
uint32_t overflow_vertex(unordered_map<uint32_t, Vertex>& ver, uint32_t s, uint32_t t)
{
    /*for (uint32_t i = 1; i < ver.size() - 1; i++) {
       if (ver[i].e_flow > 0) {
            return i;
       }
    }*/

    for (auto [key, value] : ver) {
        if (key != s && key != t) {
            if (ver[key].e_flow > 0) {
                return key;
            }
        }
    }
  
    // -1 if no overflowing Vertex
    return -1;
}
  
// Update reverse flow for flow added on ith Edge
void Network::update_reverse_edge_flow(uint32_t vert_u, uint32_t vert_v, int flow)
{
    uint32_t u = vert_v, v = vert_u;
  
    if (adjecency_map.contains(u)) {
        for (int i = 0; i < adjecency_map[u].size(); i++)
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
    for (int i = 0; i < adjecency_map[u].size(); i++)
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
            int flow = min(adjecency_map[u][i].capacity - adjecency_map[u][i].flow, ver[u].e_flow);
            
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
    int mh = numeric_limits<int>::max();
  
    // Find the adjacent with minimum height
    for (int i = 0; i < adjecency_map[u].size(); i++)
    {
        // if flow is equal to capacity then no
        // relabeling
        if (adjecency_map[u][i].flow == adjecency_map[u][i].capacity)
            continue;

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
void Network::get_min_cut(uint32_t s, uint32_t t)
{
    preflow(s);
  
    // loop untill none of the Vertex is in overflow
    while (overflow_vertex(ver, s, t) != -1) {
        uint32_t u = overflow_vertex(ver, s, t);
        
        if (!push(u)) {
            relabel(u);
        }
    }
  
    // ver.back() returns last Vertex, whose
    // e_flow will be final maximum flow

    //cout << "Maximum flow is " << ver.back().e_flow << endl;

    vector<uint32_t> s_cut;
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

    cout << "S cut is :" << endl;

    for (int i = 0; i < s_cut.size(); i++) {
        cout << s_cut[i] << ", ";
    }
    cout << endl;
}
  
// Driver program to test above functions
int main()
{
    uint32_t V = 6;
    Network g(V);
  
    // Creating above shown flow network
    g.add_edge(1, 2, 16);
    g.add_edge(1, 3, 13);
    g.add_edge(2, 3, 10);
    g.add_edge(3, 2, 4);
    g.add_edge(2, 4, 12);
    g.add_edge(3, 5, 14);
    g.add_edge(4, 3, 9);
    g.add_edge(4, 6, 20);
    g.add_edge(5, 4, 7);
    g.add_edge(5, 6, 4);
  
    // Initialize source and sink
    uint32_t s = 1, t = 6;
  
    g.get_min_cut(s, t);

    return 0;
}