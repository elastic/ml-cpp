/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */

#ifndef INCLUDED_ml_maths_CBootstrapClusterer_h
#define INCLUDED_ml_maths_CBootstrapClusterer_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CKMeansFast.h>
#include <maths/COrderings.h>
#include <maths/CPRNG.h>
#include <maths/CSetTools.h>
#include <maths/CXMeans.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/properties.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/unordered_map.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ml {
namespace maths {

//! \brief Bootstraps clustering to improve stability.
//!
//! DESCRIPTION:\n
//! This is inspired by bagged clustering
//! (see http://bioinformatics.oxfordjournals.org/content/19/9/1090.full.pdf+html).
//! However, there are a couple of significant differences:
//!   -# Standard bagged clustering approaches don't handle
//!      a variable number of clusters. This means that they
//!      are restricted in the types of clusterer to which
//!      they can be applied. The procedure we use for
//!      associating clusters in different bootstrap samples
//!      means we avoid this limitation.
//!   -# We use a non-standard mechanism for associating
//!      clusters between bootstrap samples. Rather than
//!      maximizing the overlap for a permutation of the
//!      cluster labels we connect two clusters \f$A^{(i)}\f$
//!      and \f$B^{(j)}\f$ from bootstrap samples \f$\Omega_i\f$
//!      and \f$\Omega_j\f$, respectively, if
//! <pre class="fragment">
//!   \f$\displaystyle \frac{\| A^{(i)} \cap B^{(j)} \|}{ \| \Omega_i \cup B^{(j)} \| }\f$
//! </pre>
//!      or
//! <pre class="fragment">
//!   \f$\displaystyle \frac{\| A^{(i)} \cap B^{(j)} \|}{ \| A^{(i)} \cup \Omega_j \| }\f$
//! </pre>
//!      are large. This is the overlap, or Szymkiewicz-Simpson,
//!      coefficient defined on the intersection of \f$\Omega_i\f$
//!      and \f$\Omega_j\f$. The associated clusters are densely
//!      connected components of the resulting graph.
//!
//! Once we have associated clusters we assign points based
//! on their majority vote (as with standard bagged clustering).
template<typename POINT>
class CBootstrapClusterer {
public:
    using TSizeSizePr = std::pair<std::size_t, std::size_t>;
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecItr = TSizeVec::iterator;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TSizeVecVecVec = std::vector<TSizeVecVec>;
    using TPointVec = std::vector<POINT>;
    using TPointVecVec = std::vector<TPointVec>;
    using TGraph =
        boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, double>>;
    using TVertex = typename boost::graph_traits<TGraph>::vertex_descriptor;
    using TEdge = typename boost::graph_traits<TGraph>::edge_descriptor;
    using TVertexItr = typename boost::graph_traits<TGraph>::vertex_iterator;
    using TEdgeItr = typename boost::graph_traits<TGraph>::edge_iterator;
    using TOutEdgeItr = typename boost::graph_traits<TGraph>::out_edge_iterator;
    using TAdjacencyItr = typename boost::graph_traits<TGraph>::adjacency_iterator;

public:
    CBootstrapClusterer(double overlapThreshold, double chainingFactor)
        : m_OverlapThreshold(overlapThreshold),
          m_ChainingFactor(std::max(chainingFactor, 1.0)) {}

    //! Run clustering on \p b bootstrap samples of \p points
    //! and find persistent clusters of the data.
    //!
    //! \param[in] b The number of bootstrap clusterings.
    //! \param[in] clusterer The clustering algorithm to use.
    //! \param[in] points The points to cluster.
    //! \param[out] result Filled in with the clustering.
    //!
    //! \tparam CLUSTERER Must provide a member function with
    //! signature cluster(TPointVec &, TSizeVecVec) which performs
    //! clustering. If necessary wrap up an existing clusterer
    //! with CBootstrapClustererFacade.
    template<typename CLUSTERER>
    void run(std::size_t b, CLUSTERER clusterer, TPointVec& points, TPointVecVec& result) {
        std::sort(points.begin(), points.end());
        TSizeVecVecVec bootstrapClusters;
        std::size_t n = this->bootstrapClusters(b, clusterer, points, bootstrapClusters);
        TGraph graph(n);
        this->buildClusterGraph(points, bootstrapClusters, graph);
        this->buildClusters(points, bootstrapClusters, graph, result);
    }

protected:
    using TDoubleVec = std::vector<double>;
    using TBoolVec = std::vector<bool>;
    using TSizeSizePrVec = std::vector<TSizeSizePr>;
    using TDoubleSizePr = std::pair<double, std::size_t>;
    using TDoubleSizePrVec = std::vector<TDoubleSizePr>;

    //! \brief Checks if a cluster is empty.
    struct SIsEmpty {
        bool operator()(const TPointVec& cluster) const {
            return cluster.empty();
        }
    };

    //! Check if the second elements are equal.
    struct SSecondEqual {
        bool operator()(const TDoubleSizePr& lhs, const TDoubleSizePr& rhs) const {
            return lhs.second == rhs.second;
        }
    };

    //! \brief State used for the maximum adjacency minimum cost
    //! cut search.
    struct SCutState {
        SCutState(std::size_t seed, const TGraph& graph)
            : s_V(boost::num_vertices(graph)), s_ToVisit(1, seed),
              s_Adjacency(s_V, 0), s_Cut(0.0), s_A(0) {
            this->initializeQueue();
        }

        //! Get the cost of the current cut.
        double cost() const {
            return s_Cut / static_cast<double>(s_A * (s_V - s_A));
        }

        //! Check if the vertex is to visit.
        bool toVisit(std::size_t i) const {
            return this->toVisit(s_ToVisit.size(), i);
        }

        //! Check if the vertex is to visit.
        bool toVisit(std::size_t n, std::size_t i) const {
            return std::binary_search(s_ToVisit.begin(), s_ToVisit.begin() + n, i);
        }

        //! Get the next vertex to visit.
        std::size_t next() const { return s_Queue.front().second; }

        //! Get the first right or equal vertex.
        std::size_t nextToVisit(std::size_t i) const {
            return static_cast<std::size_t>(
                std::lower_bound(s_ToVisit.begin(), s_ToVisit.end(), i) -
                s_ToVisit.begin());
        }

        //! Merge any vertices to visit after \p n.
        void mergeAfter(std::size_t n) {
            if (s_ToVisit.size() > n) {
                std::sort(s_ToVisit.begin() + n, s_ToVisit.end());
                std::inplace_merge(s_ToVisit.begin(), s_ToVisit.begin() + n,
                                   s_ToVisit.end());
            }
        }

        //! Initialize the priority queue of vertices to visit.
#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
        __attribute__((__noinline__))
#endif // defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ == 3)
        void
        initializeQueue() {
            s_Queue.clear();
            s_Queue.reserve(s_ToVisit.size());
            for (std::size_t i = 0u; i < s_ToVisit.size(); ++i) {
                s_Queue.push_back(std::make_pair(s_Adjacency[s_ToVisit[i]], i));
            }
            std::make_heap(s_Queue.begin(), s_Queue.end(), std::less<TDoubleSizePr>());
        }

        //! Pop the priority queue of vertices to visit.
        void popQueue() {
            std::pop_heap(s_Queue.begin(), s_Queue.end(), std::less<TDoubleSizePr>());
            s_Queue.pop_back();
        }

        //! The number of vertices.
        std::size_t s_V;
        //! The vertices adjacent to and outside the cut.
        TSizeVec s_ToVisit;
        //! The adjacency counts of the vertices.
        TDoubleVec s_Adjacency;
        //! Used for maintaining a priority queue of vertices to visit.
        TDoubleSizePrVec s_Queue;
        //! The current cut weight.
        double s_Cut;
        //! The current cut partitions the graph into A and V - A vertices.
        std::size_t s_A;
    };

protected:
    //! The fraction of typical cut edges to actual cut edges
    //! needed to separate a thicket.
    static const double SEPARATION_THRESHOLD;

protected:
    //! Create a collection of clusterings of \p b bootstrap
    //! samples of \p points.
    //!
    //! \param[in] b The number of bootstrap clusterings.
    //! \param[in] clusterer The clustering algorithm to use.
    //! \param[in] points The points to cluster.
    //! \param[out] result Filled in with the \p b bootstrap
    //! clusterings.
    template<typename CLUSTERER>
    std::size_t
    bootstrapClusters(std::size_t b, CLUSTERER& clusterer, TPointVec& points, TSizeVecVecVec& result) {
        std::size_t n = points.size();
        LOG_TRACE(<< "# points = " << n);

        result.clear();
        result.reserve(b);
        result.push_back(TSizeVecVec());
        clusterer.cluster(points, result.back());
        LOG_TRACE(<< "Run 1: # clusters = " << result.back().size());

        TSizeVec sampling;
        TPointVec bootstrapPoints;
        sampling.reserve(n);
        bootstrapPoints.reserve(n);

        for (std::size_t i = 1u; i < b; ++i) {
            sampling.clear();
            CSampling::uniformSample(m_Rng, 0, n, n, sampling);
            std::sort(sampling.begin(), sampling.end());
            LOG_TRACE(<< "# samples = " << sampling.size());

            bootstrapPoints.clear();
            for (std::size_t j = 0u; j < n; ++j) {
                bootstrapPoints.push_back(points[sampling[j]]);
            }

            result.push_back(TSizeVecVec());
            clusterer.cluster(bootstrapPoints, result.back());
            for (std::size_t j = 0u; j < result.back().size(); ++j) {
                for (std::size_t k = 0u; k < (result.back())[j].size(); ++k) {
                    (result.back())[j][k] = sampling[(result.back())[j][k]];
                }
            }
            LOG_TRACE(<< "Run " << i + 1 << ": # clusters = " << result.back().size());
        }

        m_Offsets.clear();
        m_Offsets.resize(result.size());
        std::size_t k = 0u;
        for (std::size_t i = 0u; i < result.size(); ++i) {
            m_Offsets[i] = k;
            k += result[i].size();
        }
        return k;
    }

    //! Build a graph of by connecting strongly similar clusters.
    //!
    //! \param[in] points The points to cluster.
    //! \param[in] bootstrapClusters The clusters of the bootstrap
    //! sampled data.
    //! \param[out] graph A graph whose vertices are the clusters
    //! in each bootstrap clustering and whose edges connect clusters
    //! which overlap significantly.
    void buildClusterGraph(const TPointVec& points,
                           TSizeVecVecVec& bootstrapClusters,
                           TGraph& graph) const {
        using TSizeSizePrUSet = boost::unordered_set<TSizeSizePr>;
        using TSizeSizePrUSetCItr = TSizeSizePrUSet::const_iterator;

        TSizeSizePrUSet edges;

        // If there are no common points between a cluster and
        // another bootstrap sampling we remember it and assign
        // edges which are consistent with its overlap with other
        // clusters.
        TSizeSizePrUSet ambiguous;

        for (std::size_t i = 0u; i < bootstrapClusters.size(); ++i) {
            for (std::size_t j = 0u; j < bootstrapClusters[i].size(); ++j) {
                std::sort(bootstrapClusters[i][j].begin(),
                          bootstrapClusters[i][j].end());
            }
        }
        TSizeVec cik;
        cik.reserve(points.size());
        TDoubleVec overlaps;
        for (std::size_t i = 0u; i < bootstrapClusters.size(); ++i) {
            for (std::size_t j = 0u; j < bootstrapClusters.size(); ++j) {
                if (i == j) {
                    continue;
                }

                for (std::size_t k = 0u; k < bootstrapClusters[i].size(); ++k) {
                    cik = bootstrapClusters[i][k];
                    cik.erase(std::unique(cik.begin(), cik.end()), cik.end());
                    double nik = static_cast<double>(cik.size());

                    overlaps.clear();
                    double sum = 0.0;

                    for (std::size_t l = 0u;
                         !cik.empty() && l < bootstrapClusters[j].size(); ++l) {
                        const TSizeVec& cjl = bootstrapClusters[j][l];
                        double o = static_cast<double>(cik.size());
                        CSetTools::inplace_set_difference(cik, cjl.begin(), cjl.end());
                        o -= static_cast<double>(cik.size());
                        o /= nik;
                        overlaps.push_back(o);
                        sum += o;
                    }

                    if (sum == 0.0) {
                        ambiguous.insert(std::make_pair(this->toVertex(i, k), j));
                    } else {
                        for (std::size_t l = 0u; l < overlaps.size(); ++l) {
                            if (overlaps[l] > m_OverlapThreshold * sum) {
                                std::size_t u = this->toVertex(i, k);
                                std::size_t v = this->toVertex(j, l);
                                if (u > v) {
                                    std::swap(u, v);
                                }
                                if (edges.insert(std::make_pair(u, v)).second) {
                                    boost::put(boost::edge_weight, graph,
                                               boost::add_edge(u, v, graph).first,
                                               std::min(m_ChainingFactor *
                                                            (overlaps[l] - m_OverlapThreshold * sum),
                                                        1.0));
                                }
                            }
                        }
                    }
                }
            }
        }
        LOG_TRACE(<< "ambiguous = " << core::CContainerPrinter::print(ambiguous));

        TDoubleSizePrVec consistent;
        for (TSizeSizePrUSetCItr i = ambiguous.begin(); i != ambiguous.end(); ++i) {
            std::size_t u = i->first;

            consistent.clear();
            TOutEdgeItr j, endj;
            for (boost::tie(j, endj) = boost::out_edges(u, graph); j != endj; ++j) {
                std::size_t v = boost::target(*j, graph);
                double weight = boost::get(boost::edge_weight, graph, *j);

                TOutEdgeItr k, endk;
                for (boost::tie(k, endk) = boost::out_edges(v, graph); k != endk; ++k) {
                    std::size_t w = boost::target(*k, graph);
                    if (this->fromVertex(w).first == i->second) {
                        consistent.push_back(std::make_pair(
                            weight * boost::get(boost::edge_weight, graph, *k), w));
                    }
                }
            }
            std::sort(consistent.begin(), consistent.end(), COrderings::SSecondLess());
            consistent.erase(std::unique(consistent.begin(), consistent.end(), SSecondEqual()),
                             consistent.end());
            LOG_TRACE(<< "consistent = " << core::CContainerPrinter::print(consistent));

            for (std::size_t k = 0u; k < consistent.size(); ++k) {
                boost::put(boost::edge_weight, graph,
                           boost::add_edge(u, consistent[k].second, graph).first,
                           consistent[k].first);
            }
        }
    }

    //! Build the clusters from the maximum connected components
    //! of \p graph.
    //!
    //! \param[in] points The points to cluster.
    //! \param[in] bootstrapClusters The bootstrap clusters of
    //! \p points.
    //! \param[in] graph The graph of overlapping clusters in
    //! \p bootstrapClusters.
    //! \param[out] result Filled in with the majority vote clusters
    //! of \p bootstrapClusters.
    void buildClusters(const TPointVec& points,
                       const TSizeVecVecVec& bootstrapClusters,
                       const TGraph& graph,
                       TPointVecVec& result) const {
        using TSizeSizeUMap = boost::unordered_map<std::size_t, std::size_t>;
        using TSizeSizeUMapCItr = TSizeSizeUMap::const_iterator;
        using TSizeSizeUMapVec = std::vector<TSizeSizeUMap>;

        // Find the maximum connected components.
        TSizeVec components(boost::num_vertices(graph));
        std::size_t n = boost::connected_components(graph, &components[0]);
        LOG_TRACE(<< "# vertices = " << components.size());
        LOG_TRACE(<< "Connected components = " << n);

        // Find components which aren't easily separable. These will
        // be the voting population.
        n = this->thickets(n, graph, components);
        LOG_TRACE(<< "thickets = " << n);

        // Build a map from voters to point indices.
        TSizeSizeUMapVec voters(n);
        for (std::size_t i = 0u; i < components.size(); ++i) {
            TSizeSizeUMap& cluster = voters[components[i]];
            const TSizeVec& vertex = this->fromVertex(bootstrapClusters, i);
            for (std::size_t j = 0u; j < vertex.size(); ++j) {
                ++cluster[vertex[j]];
            }
        }

        // Extract clusters via majority vote.
        result.clear();
        result.resize(voters.size());
        for (std::size_t i = 0u; i < points.size(); ++i) {
            std::size_t jmax = 0u;
            std::size_t cmax = 0u;
            std::size_t nmax = 0u;
            for (std::size_t j = 0u; j < n; ++j) {
                TSizeSizeUMapCItr k = voters[j].find(i);
                if (k == voters[j].end()) {
                    continue;
                }
                std::size_t c = k->second;
                std::size_t n_ = voters[j].size();
                if (COrderings::lexicographical_compare(
                        c, n_, cmax, nmax, std::greater<std::size_t>())) {
                    jmax = j;
                    cmax = c;
                    nmax = n_;
                }
            }
            if (cmax == 0) {
                LOG_ERROR(<< "Failed to find cluster for " << points[i]);
                continue;
            }

            result[jmax].push_back(points[i]);
        }

        // It is possible that after voting clusters contain
        // no points. Remove these.
        result.erase(std::remove_if(result.begin(), result.end(), SIsEmpty()),
                     result.end());
    }

    //! Identify subsets of the component of \p graph which
    //! are difficult to separate by removing edges.
    //!
    //! \param[in] n The number of components.
    //! \param[in] graph The graph for which to identify thickets.
    //! \param[in,out] components The component labels of the
    //! vertices of \p graph. Filled in with the thicket labels
    //! of \p graph.
    //! \return The number of thickets in \p graph.
    std::size_t thickets(std::size_t n, const TGraph& graph, TSizeVec& components) const {
        std::size_t V = boost::num_vertices(graph);

        TSizeVec mapping(V);
        TSizeVec inverse;
        TBoolVec parities(V);
        TGraph component(1);

        for (std::size_t i = 0u; i < n; ++i) {
            LOG_TRACE(<< "component = " << i);

            // Extract the component vertices.
            inverse.clear();
            for (std::size_t j = 0u; j < V; ++j) {
                if (components[j] == i) {
                    inverse.push_back(j);
                    mapping[j] = inverse.size() - 1;
                }
            }

            std::size_t Vi = inverse.size();

            if (Vi < 3) {
                continue;
            }

            // Build the component graph.
            this->copy(graph, mapping, inverse, component);

            // Find the partitions of the component which are difficult
            // to separate (by removing edges).
            if (this->separate(component, parities)) {
                LOG_TRACE(<< "Separated component");
                LOG_TRACE(<< "parities = "
                          << core::CContainerPrinter::print(parities.begin(),
                                                            parities.begin() + Vi));
                for (std::size_t j = 0u; j < Vi; ++j) {
                    if (parities[j]) {
                        components[inverse[j]] = n;
                    }
                }
                LOG_TRACE(<< "components = " << core::CContainerPrinter::print(components));
                ++n;
            }
        }

        return n;
    }

    //! Test to see if we should separate \p graph by a minimum cut.
    //!
    //! The idea of this test is that if there exists a cut in the
    //! graph which contains many fewer edges than we'd expect given
    //! the number of edges in the graph then we should separate
    //! the graph along this cut. This is to avoid the problem that
    //! a small number of different clusterings of the data can
    //! cause us to chain together otherwise consistently disjointly
    //! clusters.
    //!
    //! \param[in] graph The graph to separate.
    //! \param[out] result Filled in with the parity of each vertex
    //! in a cut of \p graph which minimizes the split criterion.
    //! \return True if we should split \p graph and false otherwise.
    bool separate(const TGraph& graph, TBoolVec& result) const {
        std::size_t V = boost::num_vertices(graph);
        std::size_t E = boost::num_edges(graph);

        result.assign(V, true);

        std::size_t D = V;
        for (std::size_t i = 0u; i < V; ++i) {
            D = std::min(D, boost::out_degree(i, graph));
        }
        TDoubleVec weights;
        weights.reserve(E);
        double totalWeight = 0.0;
        {
            TEdgeItr i, end;
            for (boost::tie(i, end) = boost::edges(graph); i != end; ++i) {
                double weight = boost::get(boost::edge_weight, graph, *i);
                weights.push_back(weight);
                totalWeight += weight;
            }
        }

        double p = totalWeight / static_cast<double>(V * (V - 1) / 2);
        double threshold = SEPARATION_THRESHOLD * p;

        // We can bound the ratio of the cut size to the typical
        // by noting that in most separable configuration we'd
        // remove all edges we're short of a complete graph from
        // the cut. This is a poor bound when the split is uneven,
        // i.e. unless the graph is close to complete
        // V (V - 1) / 2 - E > V - 1 so the graph could disconnected.
        // In this case, we can bound the ratio based on the minimum
        // vertex degree D for cuts in which one component contains
        // fewer than D + 1 vertices.
        double bound = std::numeric_limits<double>::max();
        std::sort(weights.begin(), weights.end());
        for (std::size_t i = 1u; i < weights.size(); ++i) {
            weights[i] += weights[i - 1];
        }
        for (std::size_t i = 1u; i <= V / 2 + 1; ++i) {
            std::size_t C = std::max(
                i * (D - std::min(D, i - 1)),
                (i * (V - i)) - std::min(i * (V - i), (V * (V - 1)) / 2 - E));
            bound = std::min(bound, weights[C] / static_cast<double>(i * (V - i)));
        }
        LOG_TRACE(<< "bound = " << bound << " threshold = " << threshold);

        if (bound >= threshold) {
            LOG_TRACE(<< "Short circuit: D = " << D << ", V = " << V << ", bound = " << bound
                      << ", threshold = " << SEPARATION_THRESHOLD * p);
            return false;
        }

        TDoubleVec seeds;
        CSampling::uniformSample(m_Rng, 0.0, 1.0, 6, seeds);

        TSizeSizePrVec cut;
        TSizeSizePrVec newCut;
        for (std::size_t i = 0u; i < seeds.size(); ++i) {
            if (cut.empty()) {
                TEdgeItr seed = boost::edges(graph).first;
                for (std::size_t j = 0u;
                     j < static_cast<std::size_t>(seeds[i] * static_cast<double>(E));
                     ++j, ++seed) {}
                cut.push_back(std::make_pair(boost::source(*seed, graph),
                                             boost::target(*seed, graph)));
            }

            double cost;
            if (this->cutSearch(cut.back().first, cut.back().second, graph,
                                threshold, cost, result)) {
                return true;
            }

            cut.pop_back();
            std::size_t n = cut.size();

            TEdgeItr j, end;
            for (boost::tie(j, end) = boost::edges(graph); j != end; ++j) {
                std::size_t u = boost::source(*j, graph);
                std::size_t v = boost::target(*j, graph);
                if (result[u] != result[v]) {
                    cut.push_back(std::make_pair(u, v));
                }
            }
            if (n > 0) {
                std::sort(cut.begin() + n, cut.end());
                newCut.clear();
                std::set_intersection(cut.begin(), cut.begin() + n, cut.begin() + n,
                                      cut.end(), std::back_inserter(newCut));
                cut.swap(newCut);
            }
        }

        return false;
    }

    //! Look for the sparsest cut of the graph including (\p u, \p v).
    //!
    //! Finding the sparsest cut is NP-hard; however, there exist
    //! effective approximate solutions. See, for example,
    //! http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.187.3506&rep=rep1&type=pdf
    //! and references therein. Here, we take a simpler to implement
    //! approach and starting with the edge (\p u, \p v), create a
    //! sequence of cuts which divide the graph into exactly two
    //! pieces using a maximum adjacency search.
    //!
    //! This is exactly how the Stoer Wagner minimum cut phase
    //! search works except for the additional constraint on
    //! connectivity (see http://e-maxx.ru/bookz/files/stoer_wagner_mincut.pdf).
    //! Note that their inductive argument that this will find the
    //! minimum cut through the last edge visited no longer works
    //! because the sparsest cut cost function is not countably
    //! additive in the edge weights in the cut. Instead, we remember
    //! the sparsest cut we find along the way.
    //!
    //! \param[in] u The start vertex of the seed edge.
    //! \param[in] v The end vertex of the seed edge.
    //! \param[in] graph The graph to separate.
    //! \param[out] cost Filled in with the lowest cost of the cut.
    //! \param[out] parities Filled in with the vertex parities in
    //! the lowest cost cut.
    //! \return True if the cut should split \p graph and false
    //! otherwise.
    bool cutSearch(std::size_t u,
                   std::size_t v,
                   const TGraph& graph,
                   double threshold,
                   double& cost,
                   TBoolVec& parities) const {
        LOG_TRACE(<< "Seed edge = (" << u << "," << v << ")");

        std::size_t V = boost::num_vertices(graph);

        parities.assign(V, true);

        SCutState state(u, graph);
        std::size_t next = state.next();
        this->visit(next, graph, parities, state);
        next = state.nextToVisit(v);
        this->visit(next, graph, parities, state);

        double lowestCost = state.cost();
        double bestCut = state.s_Cut;
        std::size_t bestA = state.s_A;
        TBoolVec best = parities;

        while (state.s_A + 1 < V) {
            if (!this->findNext(parities, graph, state)) {
                LOG_TRACE(<< "The positive subgraph is already disconnected");

                TSizeVec components;
                std::size_t c = this->positiveSubgraphConnected(graph, parities, components);
                LOG_TRACE(<< "components = " << core::CContainerPrinter::print(components));

                // Find the smallest component.
                TSizeVec sizes(c, 0);
                for (std::size_t i = 0u; i < components.size(); ++i) {
                    if (parities[i]) {
                        ++sizes[components[i]];
                    }
                }
                std::size_t smallest = static_cast<std::size_t>(
                    std::min_element(sizes.begin(), sizes.end()) - sizes.begin());
                LOG_TRACE(<< "sizes = " << core::CContainerPrinter::print(sizes));
                LOG_TRACE(<< "smallest = " << smallest);

                // Add all its vertices to the "to visit" set.
                std::size_t n = state.s_ToVisit.size();
                for (std::size_t i = 0u; i < components.size(); ++i) {
                    if (parities[i] && components[i] == smallest && !state.toVisit(i)) {
                        state.s_ToVisit.push_back(i);
                    }
                }

                state.mergeAfter(n);

                for (std::size_t i = 0u; i < components.size(); ++i) {
                    if (parities[i] && components[i] == smallest) {
                        next = state.nextToVisit(i);
                        this->visit(next, graph, parities, state);
                    }
                }
            } else {
                next = state.next();
                this->visit(next, graph, parities, state);
            }

            double cutCost = state.cost();
            if (cutCost < lowestCost) {
                lowestCost = cutCost;
                bestCut = state.s_Cut;
                bestA = state.s_A;
                best = parities;
            }
        }

        cost = lowestCost;
        parities.swap(best);

        LOG_TRACE(<< "Best cut = " << bestCut << ", |A| = " << bestA << ", |B| = " << V - bestA
                  << ", cost = " << cost << ", threshold = " << threshold);

        return cost < threshold;
    }

    //! Get the offsets for the clusterings.
    TSizeVec& offsets() { return m_Offsets; }

private:
    //! \brief A parity filter predicate which tests whether
    //! vertices and edges belong to a specified parity subgraph.
    //!
    //! This is intended for use with boost::filtered_graph.
    class CParityFilter {
    public:
        CParityFilter()
            : m_Graph(nullptr), m_Parities(nullptr), m_Parity(false) {}
        CParityFilter(const TGraph& graph, const TBoolVec& parities, bool parity)
            : m_Graph(&graph), m_Parities(&parities), m_Parity(parity) {}

        //! Check the vertex parity.
        bool operator()(const TVertex& v) const {
            return (*m_Parities)[v] == m_Parity;
        }

        //! Check the end vertices' parity.
        bool operator()(const TEdge& e) const {
            return (*m_Parities)[boost::source(e, *m_Graph)] == m_Parity &&
                   (*m_Parities)[boost::target(e, *m_Graph)] == m_Parity;
        }

    private:
        //! The graph to filter.
        const TGraph* m_Graph;
        //! The parities of the vertices of \p graph.
        const TBoolVec* m_Parities;
        //! The parity of the filtered graph.
        bool m_Parity;
    };

private:
    //! Copy the vertices in \p inverse and edges between them
    //! from \p graph into a new graph structure in \p result.
    void
    copy(const TGraph& graph, const TSizeVec& mapping, const TSizeVec& inverse, TGraph& result) const {
        result = TGraph(inverse.size());
        for (std::size_t i = 0u; i < inverse.size(); ++i) {
            TOutEdgeItr j, end;
            for (boost::tie(j, end) = boost::out_edges(inverse[i], graph); j != end; ++j) {
                std::size_t u = boost::source(*j, graph);
                std::size_t v = boost::target(*j, graph);
                if (u < v && std::binary_search(inverse.begin(), inverse.end(), v)) {
                    boost::put(boost::edge_weight, result,
                               boost::add_edge(mapping[u], mapping[v], result).first,
                               boost::get(boost::edge_weight, graph, *j));
                }
            }
        }
    }

    //! Find the next vertex to visit.
    //!
    //! This is the most adjacent vertex which doesn't disconnect
    //! the positive parity subgraph.
    bool findNext(const TBoolVec& parities, const TGraph& graph, SCutState& state) const {
        state.initializeQueue();
        TSizeVec components;
        for (std::size_t i = 0u; i < state.s_ToVisit.size(); ++i) {
            std::size_t candidate = state.next();
            std::size_t v = state.s_ToVisit[candidate];
            const_cast<TBoolVec&>(parities)[v] = false;
            bool connected =
                (this->positiveSubgraphConnected(graph, parities, components) == 1);
            const_cast<TBoolVec&>(parities)[v] = true;
            if (connected) {
                return true;
            }
            state.popQueue();
        }
        return false;
    }

    //! Visit the most adjacent vertex in the "to visit" set.
    //!
    //! This updates the "to visit" set to include newly adjacent
    //! vertices to A, the vertex adjacencies, the cut weight, and
    //! set sizes \f$|A|\f$ and \f$V - |A|\f$.
    void visit(std::size_t next, const TGraph& graph, TBoolVec& parities, SCutState& state) const {
        std::size_t u = state.s_ToVisit[next];
        LOG_TRACE(<< "Visiting " << u);

        parities[u] = false;
        state.s_ToVisit.erase(state.s_ToVisit.begin() + next);

        std::size_t n = state.s_ToVisit.size();
        TOutEdgeItr i, end;
        for (boost::tie(i, end) = boost::out_edges(u, graph); i != end; ++i) {
            double weight = boost::get(boost::edge_weight, graph, *i);
            std::size_t v = boost::target(*i, graph);
            if (parities[v]) {
                state.s_Adjacency[v] += weight;
                state.s_Cut += weight;
                if (!state.toVisit(n, v)) {
                    state.s_ToVisit.push_back(v);
                }
            } else {
                state.s_Cut -= weight;
            }
        }

        state.mergeAfter(n);
        state.s_Adjacency[u] = 0;
        ++state.s_A;
    }

    //! Check that the subgraph with true parity is connected.
    std::size_t positiveSubgraphConnected(const TGraph& graph,
                                          const TBoolVec& parities,
                                          TSizeVec& components) const {
        using TParityGraph = boost::filtered_graph<TGraph, CParityFilter, CParityFilter>;
        CParityFilter parityFilter(graph, parities, true);
        TParityGraph parityGraph(graph, parityFilter, parityFilter);
        components.resize(boost::num_vertices(graph));
        return boost::connected_components(parityGraph, &components[0]);
    }

    //! Extract the vertex for the \p j'th cluster of the
    //! \p i'th bootstrap clustering.
    std::size_t toVertex(std::size_t i, std::size_t j) const {
        return m_Offsets[i] + j;
    }

    //! Extract the clustering and cluster from the vertex
    //! representation \p v.
    TSizeSizePr fromVertex(std::size_t v) const {
        std::size_t i = static_cast<std::size_t>(
                            std::upper_bound(m_Offsets.begin(), m_Offsets.end(), v) -
                            m_Offsets.begin()) -
                        1;
        return std::make_pair(i, v - m_Offsets[i]);
    }

    //! Extract the cluster corresponding to the \p v'th vertex
    //! of the cluster graph.
    const TSizeVec& fromVertex(const TSizeVecVecVec& clusters, std::size_t v) const {
        TSizeSizePr ij = fromVertex(v);
        return clusters[ij.first][ij.second];
    }

private:
    //! The random number generator.
    mutable CPRNG::CXorShift1024Mult m_Rng;

    //! The threshold in the similarity measure for which we will
    //! consider joining clusters.
    double m_OverlapThreshold;

    //! The amount overlap between clusters causes them to chain
    //! together.
    double m_ChainingFactor;

    //! A flat encoding of the vertices in each clustering.
    //!
    //! In particular, the start of the i'th clustering clusters
    //! is encoded by the i'th element.
    TSizeVec m_Offsets;
};

template<typename POINT>
const double CBootstrapClusterer<POINT>::SEPARATION_THRESHOLD(0.1);

//! \brief Extracts the clusters in canonical form (by index into
//! the point vector) for the facade implementations.
template<typename POINT>
class CBootstrapClustererFacadeExtractClusters {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TPointVec = std::vector<POINT>;
    using TPointVecCItr = typename TPointVec::const_iterator;

public:
    //! Compute the cluster of each point in \p points.
    //!
    //! \param[in] points The ordered points to cluster.
    //! \param[in] clusters The clustering of \p points.
    //! \param[out] result Filled in with the clustering of the
    //! indexes of \p points.
    template<typename CLUSTERS>
    void extract(const TPointVec& points, const CLUSTERS& clusters, TSizeVecVec& result) {

        result.resize(clusters.size());

        for (std::size_t i = 0u; i < clusters.size(); ++i) {
            const TPointVec& clusterPoints = clusters[i];

            result[i].clear();
            result[i].reserve(clusterPoints.size());

            for (std::size_t j = 0u; j < clusterPoints.size(); ++j) {
                std::size_t k = points.size();
                for (TPointVecCItr l = this->begin(points, clusterPoints[j]),
                                   end = this->end(points, clusterPoints[j]);
                     l != end; ++l) {
                    if (*l == clusterPoints[j]) {
                        k = static_cast<std::size_t>(l - points.begin());
                        break;
                    }
                }

                if (k == points.size()) {
                    LOG_ERROR(<< "Didn't find point " << clusterPoints[j]);
                    continue;
                }

                result[i].push_back(k);
            }
        }
    }

private:
    //! Get the first point equal or right of \p x.
    TPointVecCItr begin(const TPointVec& points, const POINT& x) const {
        return std::lower_bound(points.begin(), points.end(), x);
    }

    //! Get the first point right of \p x.
    TPointVecCItr end(const TPointVec& points, const POINT& x) const {
        return std::upper_bound(points.begin(), points.end(), x);
    }
};

//! \brief Adapts clustering implementations for use by the bootstrap
//! clusterer.
template<typename CLUSTERER>
class CBootstrapClustererFacade {};

//! \brief Adapts the x-means implementation for use by the bootstrap
//! clusterer.
template<typename POINT, typename COST>
class CBootstrapClustererFacade<CXMeans<POINT, COST>>
    : private CBootstrapClustererFacadeExtractClusters<POINT> {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TPointVec = std::vector<POINT>;

public:
    CBootstrapClustererFacade(const CXMeans<POINT, COST>& xmeans,
                              std::size_t improveParamsKmeansIterations,
                              std::size_t improveStructureClusterSeeds,
                              std::size_t improveStructureKmeansIterations)
        : m_Xmeans(xmeans),
          m_ImproveParamsKmeansIterations(improveParamsKmeansIterations),
          m_ImproveStructureClusterSeeds(improveStructureClusterSeeds),
          m_ImproveStructureKmeansIterations(improveStructureKmeansIterations) {}

    //! \note Assumes \p points are sorted.
    void cluster(const TPointVec& points, TSizeVecVec& result) {
        using TPointVecCRef = boost::reference_wrapper<const TPointVec>;
        using TPointVecCRefVec = std::vector<TPointVecCRef>;

        // Initialize
        TPointVec tmp(points);
        m_Xmeans.setPoints(tmp);

        // Run
        m_Xmeans.run(m_ImproveParamsKmeansIterations, m_ImproveStructureClusterSeeds,
                     m_ImproveStructureKmeansIterations);

        // Extract
        TPointVecCRefVec clusterPoints;
        for (std::size_t i = 0u; i < m_Xmeans.clusters().size(); ++i) {
            clusterPoints.push_back(boost::cref(m_Xmeans.clusters()[i].points()));
        }
        this->extract(points, clusterPoints, result);
    }

private:
    //! The x-means implementation.
    CXMeans<POINT, COST> m_Xmeans;
    //! The number of iterations to use in k-means for a single round
    //! of improve parameters.
    std::size_t m_ImproveParamsKmeansIterations;
    //! The number of random seeds to try when initializing k-means
    //! for a single round of improve structure.
    std::size_t m_ImproveStructureClusterSeeds;
    //! The number of iterations to use in k-means for a single round
    //! of improve structure.
    std::size_t m_ImproveStructureKmeansIterations;
};

//! \brief Adapts the x-means implementation for use by the bootstrap
//! clusterer.
template<typename POINT>
class CBootstrapClustererFacade<CKMeansFast<POINT>>
    : private CBootstrapClustererFacadeExtractClusters<POINT> {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TSizeVecVec = std::vector<TSizeVec>;
    using TPointVec = std::vector<POINT>;

public:
    CBootstrapClustererFacade(const CKMeansFast<POINT>& kmeans, std::size_t k, std::size_t maxIterations)
        : m_Kmeans(kmeans), m_K(k), m_MaxIterations(maxIterations) {}

    //! \note Assumes \p points are sorted.
    void cluster(const TPointVec& points, TSizeVecVec& result) {
        using TPointVecVec = std::vector<TPointVec>;

        // Initialize
        TPointVec tmp(points);
        m_Kmeans.setPoints(tmp);
        TPointVec centres;
        CKMeansPlusPlusInitialization<POINT, CPRNG::CXorShift1024Mult> seedCentres(m_Rng);
        seedCentres.run(points, m_K, centres);
        m_Kmeans.setCentres(centres);

        // Run
        m_Kmeans.run(m_MaxIterations);

        // Extract
        TPointVecVec clusterPoints;
        m_Kmeans.closestPoints(clusterPoints);
        this->extract(points, clusterPoints, result);
    }

private:
    //! The random number generator.
    CPRNG::CXorShift1024Mult m_Rng;
    //! The k-means implementation.
    CKMeansFast<POINT> m_Kmeans;
    //! The number of clusters to use.
    std::size_t m_K;
    //! The number of iterations to use in k-means.
    std::size_t m_MaxIterations;
};

//! Cluster \p points using \p B bootstrap samples using x-means.
//!
//! \param[in] points The points to cluster.
//! \param[in] B The number of bootstrap (with replacement) samples
//! of \p points to use.
//! \param[in] xmeans Used to cluster each sample.
//! \param[in] improveParamsKmeansIterations The number of iterations
//! of Lloyd's algorithm to use in k-means for a single round of improve
//! parameters.
//! \param[in] improveStructureClusterSeeds The number of random seeds
//! to try when initializing k-means for a single round of improve
//! structure.
//! \param[in] improveStructureKmeansIterations The number of iterations
//! of Lloyd's algorithm to use in k-means for a single round of improve
//! structure.
//! \param[in] overlapThreshold The similarity in terms of the overlap
//! coefficient at which we start to identify two clusters.
//! \param[in] chainingFactor The degree to which we will chain similar
//! clusters.
//! \param[out] result Filled in with the clustering of \p points.
template<typename POINT, typename COST>
void bootstrapCluster(std::vector<POINT>& points,
                      std::size_t B,
                      const CXMeans<POINT, COST>& xmeans,
                      std::size_t improveParamsKmeansIterations,
                      std::size_t improveStructureClusterSeeds,
                      std::size_t improveStructureKmeansIterations,
                      double overlapThreshold,
                      double chainingFactor,
                      std::vector<std::vector<POINT>>& result) {
    CBootstrapClustererFacade<CXMeans<POINT, COST>> clusterer(
        xmeans, improveParamsKmeansIterations, improveStructureClusterSeeds,
        improveStructureKmeansIterations);
    CBootstrapClusterer<POINT> bootstrapClusterer(overlapThreshold, chainingFactor);
    bootstrapClusterer.run(B, clusterer, points, result);
}

//! Cluster \p points using \p B bootstrap samples using k-means.
template<typename POINT>
void bootstrapCluster(std::vector<POINT>& points,
                      std::size_t B,
                      const CKMeansFast<POINT>& kmeans,
                      std::size_t k,
                      std::size_t maxIterations,
                      double overlapThreshold,
                      double chainingFactor,
                      std::vector<std::vector<POINT>>& result) {
    CBootstrapClustererFacade<CKMeansFast<POINT>> clusterer(kmeans, k, maxIterations);
    CBootstrapClusterer<POINT> bootstrapClusterer(overlapThreshold, chainingFactor);
    bootstrapClusterer.run(B, clusterer, points, result);
}
}
}

#endif // INCLUDED_ml_maths_CBootstrapClusterer_h
