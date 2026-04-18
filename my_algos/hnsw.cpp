#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
using namespace std;

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
typedef vector<float> VectorX;

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------
struct Node {
    int id;
    VectorX data;
    // connections[level] = list of neighbor Node IDs at that level
    vector<vector<int>> connections;
    int max_level;

    Node(int _id, const VectorX& _data, int _max_level)
        : id(_id), data(_data), max_level(_max_level) {
        connections.resize(max_level + 1);
    }
};

// ---------------------------------------------------------------------------
// HNSW
// ---------------------------------------------------------------------------
class HNSW {
private:
    vector<Node*> nodes;
    int enter_point;   // ID of the entry-point node (lives on max_level)
    int max_level;     // Highest level currently in the graph

    // ---- Hyper-parameters ------------------------------------------------
    int   M              = 16;    // Max connections per layer (except layer 0)
    int   M_max0         = 32;    // Max connections at layer 0
    int   ef_construction = 200;  // Size of dynamic candidate list during build
    float m_L;                    // Level normalisation factor = 1 / ln(M)

    mt19937 rng;
    uniform_real_distribution<double> unif;

    // ---- Helpers ---------------------------------------------------------

    // Squared Euclidean distance (no sqrt needed for comparisons)
    float distance(const VectorX& a, const VectorX& b) const {
        float d = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            d += diff * diff;
        }
        return d;
    }

    // Draw a random insertion level
    int random_level() {
        return static_cast<int>(floor(-m_L * log(unif(rng))));
    }

    // -----------------------------------------------------------------------
    // search_layer
    //   Greedy best-first search at a single layer.
    //   Returns a max-heap of (dist, id) with up to `ef` entries.
    // -----------------------------------------------------------------------
    priority_queue<pair<float,int>> search_layer(
        const VectorX& query,
        int ep_id,
        int ef,
        int layer) const
    {
        // min-heap: candidates to explore
        priority_queue<pair<float,int>,
                       vector<pair<float,int>>,
                       greater<pair<float,int>>> candidates;

        // max-heap: best `ef` found so far
        priority_queue<pair<float,int>> W;

        unordered_set<int> visited;
        visited.reserve(ef * 4);

        float d_ep = distance(query, nodes[ep_id]->data);
        candidates.push({d_ep, ep_id});
        W.push({d_ep, ep_id});
        visited.insert(ep_id);

        while (!candidates.empty()) {
            auto [d_c, c_id] = candidates.top();
            candidates.pop();

            // Pruning: if closest candidate is farther than worst in W, stop
            if (d_c > W.top().first) break;

            for (int nb : nodes[c_id]->connections[layer]) {
                if (visited.insert(nb).second) {   // not yet visited
                    float d_nb = distance(query, nodes[nb]->data);
                    if ((int)W.size() < ef || d_nb < W.top().first) {
                        candidates.push({d_nb, nb});
                        W.push({d_nb, nb});
                        if ((int)W.size() > ef) W.pop();
                    }
                }
            }
        }
        return W;
    }

    // -----------------------------------------------------------------------
    // select_neighbors_heuristic
    //   HNSW paper Algorithm 4: keeps up to `M_max` neighbours that together
    //   form a diverse, well-connected set rather than just the raw nearest.
    // -----------------------------------------------------------------------
    vector<int> select_neighbors_heuristic(
        const VectorX& query,
        priority_queue<pair<float,int>>& candidates,   // max-heap (dist, id)
        int M_max) const
    {
        // Convert max-heap → sorted ascending by distance
        vector<pair<float,int>> cands;
        cands.reserve(candidates.size());
        while (!candidates.empty()) {
            cands.push_back(candidates.top());
            candidates.pop();
        }
        sort(cands.begin(), cands.end());   // nearest first

        vector<int> result;
        result.reserve(M_max);

        for (auto& [d_e, e_id] : cands) {
            if ((int)result.size() >= M_max) break;

            // Keep e if it is closer to query than to every already-selected nb
            bool keep = true;
            for (int r_id : result) {
                if (distance(nodes[e_id]->data, nodes[r_id]->data) < d_e) {
                    keep = false;
                    break;
                }
            }
            if (keep) result.push_back(e_id);
        }
        return result;
    }

    // -----------------------------------------------------------------------
    // connect  — add a bidirectional edge and prune if needed
    // -----------------------------------------------------------------------
    void connect(int a, int b, int layer) {
        int limit = (layer == 0) ? M_max0 : M;

        // a → b
        nodes[a]->connections[layer].push_back(b);
        // b → a
        nodes[b]->connections[layer].push_back(a);

        // If b now has too many neighbours, prune
        auto& nb_b = nodes[b]->connections[layer];
        if ((int)nb_b.size() > limit) {
            // Rebuild as a candidate heap and re-select
            priority_queue<pair<float,int>> tmp;
            for (int nb : nb_b)
                tmp.push({distance(nodes[b]->data, nodes[nb]->data), nb});
            nb_b = select_neighbors_heuristic(nodes[b]->data, tmp, limit);
        }

        // If a now has too many neighbours, prune
        auto& nb_a = nodes[a]->connections[layer];
        if ((int)nb_a.size() > limit) {
            priority_queue<pair<float,int>> tmp;
            for (int nb : nb_a)
                tmp.push({distance(nodes[a]->data, nodes[nb]->data), nb});
            nb_a = select_neighbors_heuristic(nodes[a]->data, tmp, limit);
        }
    }

public:
    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    explicit HNSW(int M_ = 16, int ef_construction_ = 200, unsigned seed = 42)
        : M(M_), M_max0(2 * M_), ef_construction(ef_construction_),
          enter_point(-1), max_level(-1),
          m_L(1.0f / std::log(static_cast<float>(M_))),
          rng(seed), unif(0.0, 1.0)
    {}

    ~HNSW() { for (Node* n : nodes) delete n; }

    // -----------------------------------------------------------------------
    // insert
    // -----------------------------------------------------------------------
    void insert(const VectorX& vec) {
        int id = static_cast<int>(nodes.size());
        int l  = random_level();

        Node* new_node = new Node(id, vec, l);
        nodes.push_back(new_node);

        // First node ever
        if (enter_point == -1) {
            enter_point = id;
            max_level   = l;
            return;
        }

        int curr_ep = enter_point;

        // Phase 1: Greedy descent from max_level down to l+1 (ef = 1)
        for (int lc = max_level; lc > l; --lc) {
            auto W = search_layer(vec, curr_ep, 1, lc);
            curr_ep = W.top().second;
        }

        // Phase 2: Insert at levels min(max_level, l) down to 0
        for (int lc = min(max_level, l); lc >= 0; --lc) {
            auto W = search_layer(vec, curr_ep, ef_construction, lc);

            // Update entry point for next (lower) layer
            curr_ep = W.top().second;

            // Select M best neighbours via heuristic
            int M_lc = (lc == 0) ? M_max0 : M;
            vector<int> neighbours = select_neighbors_heuristic(vec, W, M_lc);

            for (int nb : neighbours)
                connect(id, nb, lc);
        }

        // Promote entry point if new node lives on a higher level
        if (l > max_level) {
            max_level   = l;
            enter_point = id;
        }
    }

    // -----------------------------------------------------------------------
    // knn_search
    //   Returns the k nearest neighbours of `query` as (dist, id) pairs,
    //   sorted nearest-first.  ef controls recall (higher = better but slower).
    // -----------------------------------------------------------------------
    vector<pair<float,int>> knn_search(
        const VectorX& query,
        int k,
        int ef = -1) const
    {
        if (nodes.empty()) return {};
        if (ef < k) ef = k;        // ef must be >= k

        int curr_ep = enter_point;

        // Phase 1: descend to level 1 with ef = 1
        for (int lc = max_level; lc > 0; --lc) {
            auto W = search_layer(query, curr_ep, 1, lc);
            curr_ep = W.top().second;
        }

        // Phase 2: search level 0 with full ef
        auto W = search_layer(query, curr_ep, ef, 0);

        // Convert max-heap → sorted ascending, keep top k
        vector<pair<float,int>> result;
        result.reserve(W.size());
        while (!W.empty()) { result.push_back(W.top()); W.pop(); }
        sort(result.begin(), result.end());
        if ((int)result.size() > k) result.resize(k);
        return result;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    int size()      const { return static_cast<int>(nodes.size()); }
    int num_levels()const { return max_level + 1; }
};

// ---------------------------------------------------------------------------
// Utility: build a random dataset and run a brute-force k-NN for comparison
// ---------------------------------------------------------------------------
static float brute_force_dist(const VectorX& a, const VectorX& b) {
    float d = 0;
    for (size_t i = 0; i < a.size(); ++i) { float diff = a[i]-b[i]; d += diff*diff; }
    return d;
}

static vector<pair<float,int>> brute_force_knn(
    const vector<VectorX>& dataset,
    const VectorX& query, int k)
{
    vector<pair<float,int>> all;
    all.reserve(dataset.size());
    for (int i = 0; i < (int)dataset.size(); ++i)
        all.push_back({brute_force_dist(query, dataset[i]), i});
    sort(all.begin(), all.end());
    if ((int)all.size() > k) all.resize(k);
    return all;
}

// ---------------------------------------------------------------------------
// main — demo
// ---------------------------------------------------------------------------
int main() {
    // ---- Parameters --------------------------------------------------------
    const int N    = 2000;   // number of vectors to index
    const int DIM  = 64;     // dimensionality
    const int K    = 10;     // neighbours to retrieve
    const int EF   = 50;     // search ef (higher → better recall)
    const int M    = 16;     // HNSW M
    const int EFC  = 200;    // ef_construction

    mt19937 gen(123);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // ---- Build dataset -----------------------------------------------------
    cout << "Generating " << N << " random " << DIM << "-d vectors...\n";
    vector<VectorX> dataset(N, VectorX(DIM));
    for (auto& v : dataset)
        for (auto& x : v) x = dist(gen);

    // ---- Build HNSW index --------------------------------------------------
    cout << "Building HNSW index (M=" << M << ", ef_construction=" << EFC << ")...\n";
    HNSW index(M, EFC, /*seed=*/42);
    for (const auto& v : dataset) index.insert(v);
    cout << "Index built. Nodes=" << index.size()
         << "  Levels=" << index.num_levels() << "\n\n";

    // ---- Query --------------------------------------------------------------
    VectorX query(DIM);
    for (auto& x : query) x = dist(gen);

    cout << "k-NN query (k=" << K << ", ef=" << EF << "):\n";
    auto hnsw_result = index.knn_search(query, K, EF);
    auto bf_result   = brute_force_knn(dataset, query, K);

    // ---- Recall@K ----------------------------------------------------------
    unordered_set<int> bf_ids;
    for (auto& [d, id] : bf_result) bf_ids.insert(id);

    int hits = 0;
    cout << "  HNSW result (id, sq-dist):\n";
    for (auto& [d, id] : hnsw_result) {
        bool correct = bf_ids.count(id);
        if (correct) ++hits;
        cout << "    id=" << id << "  dist^2=" << d
             << (correct ? "  [correct]" : "  [missed]") << "\n";
    }
    double recall = 100.0 * hits / K;
    cout << "\nRecall@" << K << " = " << recall << "%\n";

    return 0;
}