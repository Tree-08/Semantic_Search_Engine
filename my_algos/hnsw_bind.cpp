#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// ---- Paste / include your HNSW implementation ----------------------------
// Remove the `main()` function from hnsw.cpp before including, OR just
// keep hnsw.cpp as a header-only file by adding #pragma once at the top.
// Easiest: copy everything EXCEPT main() into hnsw_core.hpp and include it:
//   #include "hnsw_core.hpp"
//
// For a self-contained build, the full HNSW class is reproduced below.
// If you already have the class in a separate header, delete this block
// and replace with:  #include "hnsw_core.hpp"
// --------------------------------------------------------------------------

#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>
using namespace std;
typedef vector<float> VectorX;

struct Node {
    int id;
    VectorX data;
    vector<vector<int>> connections;
    int max_level;
    Node(int _id, const VectorX& _data, int _max_level)
        : id(_id), data(_data), max_level(_max_level) {
        connections.resize(max_level + 1);
    }
};

class HNSW {
private:
    vector<Node*> nodes;
    int enter_point;
    int max_level;
    int   M              = 16;
    int   M_max0         = 32;
    int   ef_construction = 200;
    float m_L;
    mt19937 rng;
    uniform_real_distribution<double> unif;

    float distance(const VectorX& a, const VectorX& b) const {
        float d = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) { float diff = a[i]-b[i]; d += diff*diff; }
        return d;
    }
    int random_level() {
        return static_cast<int>(floor(-m_L * log(unif(rng))));
    }
    priority_queue<pair<float,int>> search_layer(
        const VectorX& query, int ep_id, int ef, int layer) const
    {
        priority_queue<pair<float,int>,vector<pair<float,int>>,greater<pair<float,int>>> candidates;
        priority_queue<pair<float,int>> W;
        unordered_set<int> visited; visited.reserve(ef * 4);
        float d_ep = distance(query, nodes[ep_id]->data);
        candidates.push({d_ep, ep_id}); W.push({d_ep, ep_id}); visited.insert(ep_id);
        while (!candidates.empty()) {
            auto [d_c, c_id] = candidates.top(); candidates.pop();
            if (d_c > W.top().first) break;
            for (int nb : nodes[c_id]->connections[layer]) {
                if (visited.insert(nb).second) {
                    float d_nb = distance(query, nodes[nb]->data);
                    if ((int)W.size() < ef || d_nb < W.top().first) {
                        candidates.push({d_nb, nb}); W.push({d_nb, nb});
                        if ((int)W.size() > ef) W.pop();
                    }
                }
            }
        }
        return W;
    }
    vector<int> select_neighbors_heuristic(
        const VectorX& query, priority_queue<pair<float,int>>& candidates, int M_max) const
    {
        vector<pair<float,int>> cands;
        cands.reserve(candidates.size());
        while (!candidates.empty()) { cands.push_back(candidates.top()); candidates.pop(); }
        sort(cands.begin(), cands.end());
        vector<int> result; result.reserve(M_max);
        for (auto& [d_e, e_id] : cands) {
            if ((int)result.size() >= M_max) break;
            bool keep = true;
            for (int r_id : result)
                if (distance(nodes[e_id]->data, nodes[r_id]->data) < d_e) { keep = false; break; }
            if (keep) result.push_back(e_id);
        }
        return result;
    }
    void connect(int a, int b, int layer) {
        int limit = (layer == 0) ? M_max0 : M;
        nodes[a]->connections[layer].push_back(b);
        nodes[b]->connections[layer].push_back(a);
        auto& nb_b = nodes[b]->connections[layer];
        if ((int)nb_b.size() > limit) {
            priority_queue<pair<float,int>> tmp;
            for (int nb : nb_b) tmp.push({distance(nodes[b]->data, nodes[nb]->data), nb});
            nb_b = select_neighbors_heuristic(nodes[b]->data, tmp, limit);
        }
        auto& nb_a = nodes[a]->connections[layer];
        if ((int)nb_a.size() > limit) {
            priority_queue<pair<float,int>> tmp;
            for (int nb : nb_a) tmp.push({distance(nodes[a]->data, nodes[nb]->data), nb});
            nb_a = select_neighbors_heuristic(nodes[a]->data, tmp, limit);
        }
    }

public:
    explicit HNSW(int M_ = 16, int ef_construction_ = 200, unsigned seed = 42)
        : M(M_), M_max0(2 * M_), ef_construction(ef_construction_),
          enter_point(-1), max_level(-1),
          m_L(1.0f / std::log(static_cast<float>(M_))),
          rng(seed), unif(0.0, 1.0) {}
    ~HNSW() { for (Node* n : nodes) delete n; }

    void insert(const VectorX& vec) {
        int id = static_cast<int>(nodes.size());
        int l  = random_level();
        Node* new_node = new Node(id, vec, l);
        nodes.push_back(new_node);
        if (enter_point == -1) { enter_point = id; max_level = l; return; }
        int curr_ep = enter_point;
        for (int lc = max_level; lc > l; --lc) {
            auto W = search_layer(vec, curr_ep, 1, lc); curr_ep = W.top().second;
        }
        for (int lc = min(max_level, l); lc >= 0; --lc) {
            auto W = search_layer(vec, curr_ep, ef_construction, lc);
            curr_ep = W.top().second;
            int M_lc = (lc == 0) ? M_max0 : M;
            vector<int> neighbours = select_neighbors_heuristic(vec, W, M_lc);
            for (int nb : neighbours) connect(id, nb, lc);
        }
        if (l > max_level) { max_level = l; enter_point = id; }
    }

    vector<pair<float,int>> knn_search(const VectorX& query, int k, int ef = -1) const {
        if (nodes.empty()) return {};
        if (ef < k) ef = k;
        int curr_ep = enter_point;
        for (int lc = max_level; lc > 0; --lc) {
            auto W = search_layer(query, curr_ep, 1, lc); curr_ep = W.top().second;
        }
        auto W = search_layer(query, curr_ep, ef, 0);
        vector<pair<float,int>> result;
        result.reserve(W.size());
        while (!W.empty()) { result.push_back(W.top()); W.pop(); }
        sort(result.begin(), result.end());
        if ((int)result.size() > k) result.resize(k);
        return result;
    }

    int size()       const { return static_cast<int>(nodes.size()); }
    int num_levels() const { return max_level + 1; }
};

// ---- pybind11 bindings ---------------------------------------------------
namespace py = pybind11;

PYBIND11_MODULE(hnsw_index, m) {
    m.doc() = "HNSW approximate nearest-neighbour index";

    py::class_<HNSW>(m, "HNSWIndex")
        .def(py::init<int, int, unsigned>(),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("seed") = 42,
             R"doc(
Create an HNSW index.

Parameters
----------
M : int
    Max connections per layer (default 16). Higher = better recall, more RAM.
ef_construction : int
    Candidate list size during build (default 200). Higher = better graph quality.
seed : int
    RNG seed for reproducibility (default 42).
)doc")

        // ------------------------------------------------------------------
        // insert: accepts a 1-D numpy float32 array
        // ------------------------------------------------------------------
        .def("insert",
             [](HNSW& self, py::array_t<float, py::array::c_style | py::array::forcecast> vec) {
                 auto buf = vec.request();
                 if (buf.ndim != 1)
                     throw std::runtime_error("insert() expects a 1-D float32 array");
                 VectorX v(static_cast<float*>(buf.ptr),
                           static_cast<float*>(buf.ptr) + buf.shape[0]);
                 self.insert(v);
             },
             py::arg("vec"),
             "Insert a single float32 vector (shape: [dim]).")

        // ------------------------------------------------------------------
        // insert_batch: accepts a 2-D numpy float32 array (N, dim)
        // ------------------------------------------------------------------
        .def("insert_batch",
             [](HNSW& self, py::array_t<float, py::array::c_style | py::array::forcecast> mat) {
                 auto buf = mat.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("insert_batch() expects a 2-D float32 array (N, dim)");
                 int n   = static_cast<int>(buf.shape[0]);
                 int dim = static_cast<int>(buf.shape[1]);
                 float* ptr = static_cast<float*>(buf.ptr);
                 for (int i = 0; i < n; ++i) {
                     VectorX v(ptr + i * dim, ptr + i * dim + dim);
                     self.insert(v);
                 }
             },
             py::arg("matrix"),
             "Insert all rows of a float32 matrix (shape: [N, dim]) in one call.")

        // ------------------------------------------------------------------
        // search: returns (distances, indices) as two numpy arrays
        // ------------------------------------------------------------------
        .def("search",
             [](const HNSW& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> vec,
                int k,
                int ef) -> py::tuple {
                 auto buf = vec.request();
                 if (buf.ndim != 1)
                     throw std::runtime_error("search() expects a 1-D float32 array");
                 VectorX q(static_cast<float*>(buf.ptr),
                           static_cast<float*>(buf.ptr) + buf.shape[0]);
                 auto results = self.knn_search(q, k, ef);

                 py::array_t<float> dists(results.size());
                 py::array_t<int>   ids  (results.size());
                 auto d_buf = dists.mutable_unchecked<1>();
                 auto i_buf = ids.mutable_unchecked<1>();
                 for (size_t i = 0; i < results.size(); ++i) {
                     d_buf(i) = results[i].first;
                     i_buf(i) = results[i].second;
                 }
                 return py::make_tuple(dists, ids);
             },
             py::arg("vec"),
             py::arg("k"),
             py::arg("ef") = -1,
             R"doc(
Find the k nearest neighbours of vec.

Parameters
----------
vec : np.ndarray, dtype=float32, shape (dim,)
    Query vector.
k : int
    Number of neighbours to return.
ef : int, optional
    Search candidate list size. Defaults to max(k, 50).
    Higher ef → better recall, slower search.

Returns
-------
distances : np.ndarray, shape (k,)
    Squared Euclidean distances, sorted ascending.
indices : np.ndarray, shape (k,)
    Node IDs corresponding to each distance.
)doc")

        // ------------------------------------------------------------------
        // search_batch: query many vectors at once
        // ------------------------------------------------------------------
        .def("search_batch",
             [](const HNSW& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> mat,
                int k,
                int ef) -> py::tuple {
                 auto buf = mat.request();
                 if (buf.ndim != 2)
                     throw std::runtime_error("search_batch() expects a 2-D float32 array (N, dim)");
                 int n   = static_cast<int>(buf.shape[0]);
                 int dim = static_cast<int>(buf.shape[1]);
                 float* ptr = static_cast<float*>(buf.ptr);

                 py::array_t<float> all_dists({n, k});
                 py::array_t<int>   all_ids  ({n, k});
                 auto d_buf = all_dists.mutable_unchecked<2>();
                 auto i_buf = all_ids.mutable_unchecked<2>();

                 for (int qi = 0; qi < n; ++qi) {
                     VectorX q(ptr + qi * dim, ptr + qi * dim + dim);
                     auto results = self.knn_search(q, k, ef);
                     for (int j = 0; j < (int)results.size(); ++j) {
                         d_buf(qi, j) = results[j].first;
                         i_buf(qi, j) = results[j].second;
                     }
                     // pad with -1 / inf if fewer than k results
                     for (int j = (int)results.size(); j < k; ++j) {
                         d_buf(qi, j) = std::numeric_limits<float>::infinity();
                         i_buf(qi, j) = -1;
                     }
                 }
                 return py::make_tuple(all_dists, all_ids);
             },
             py::arg("matrix"),
             py::arg("k"),
             py::arg("ef") = -1,
             R"doc(
Batch search. Query every row of matrix.

Returns
-------
distances : np.ndarray, shape (N, k)
indices   : np.ndarray, shape (N, k)
)doc")

        .def("__len__",  &HNSW::size)
        .def("__repr__", [](const HNSW& self) {
            return "<HNSWIndex size=" + std::to_string(self.size()) +
                   " levels=" + std::to_string(self.num_levels()) + ">";
        })
        .def_property_readonly("size",       &HNSW::size)
        .def_property_readonly("num_levels", &HNSW::num_levels);
}