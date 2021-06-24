/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>

#include <maths/CBootstrapClusterer.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraTools.h>

#include <test/BoostTestCloseAbsolute.h>
#include <test/CRandomNumbers.h>

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <vector>

using TVector2 = ml::maths::CVectorNx1<double, 2>;
struct SVector2Hash {
    std::size_t operator()(const TVector2& x) const {
        return static_cast<std::size_t>(x.checksum());
    }
};
using TVector2SizeUMap = boost::unordered_map<TVector2, std::size_t, SVector2Hash>;

BOOST_TEST_DONT_PRINT_LOG_VALUE(TVector2SizeUMap::iterator)

BOOST_AUTO_TEST_SUITE(CBootstrapClustererTest)

using namespace ml;

namespace {

using TBoolVec = std::vector<bool>;
using TDoubleVec = std::vector<double>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TVector2Vec = std::vector<TVector2>;
using TVector2VecVec = std::vector<TVector2Vec>;
using TMatrix2 = maths::CSymmetricMatrixNxN<double, 2>;
using TMatrix2Vec = std::vector<TMatrix2>;
using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;

template<typename POINT>
class CBootstrapClustererForTest : public maths::CBootstrapClusterer<POINT> {
public:
    using TBoolVec = typename maths::CBootstrapClusterer<POINT>::TBoolVec;
    using TSizeVec = typename maths::CBootstrapClusterer<POINT>::TSizeVec;
    using TSizeVecVecVec = typename maths::CBootstrapClusterer<POINT>::TSizeVecVecVec;
    using TPointVec = typename maths::CBootstrapClusterer<POINT>::TPointVec;
    using TGraph = typename maths::CBootstrapClusterer<POINT>::TGraph;

public:
    CBootstrapClustererForTest(double overlapThreshold, double chainingFactor)
        : maths::CBootstrapClusterer<POINT>(overlapThreshold, chainingFactor) {}

    void buildClusterGraph(TSizeVecVecVec& bootstrapClusters, TGraph& graph) const {
        TPointVec dummy(1); // only used for reserving memory.
        this->maths::CBootstrapClusterer<POINT>::buildClusterGraph(
            dummy, bootstrapClusters, graph);
    }

    std::size_t thickets(std::size_t n, const TGraph& graph, TSizeVec& components) const {
        return this->maths::CBootstrapClusterer<POINT>::thickets(n, graph, components);
    }

    bool separate(TGraph& graph, TBoolVec& parity) const {
        return this->maths::CBootstrapClusterer<POINT>::separate(graph, parity);
    }

    bool cutSearch(std::size_t u,
                   std::size_t v,
                   const TGraph& graph,
                   double threshold,
                   double& cost,
                   TBoolVec& parities) const {
        return this->maths::CBootstrapClusterer<POINT>::cutSearch(
            u, v, graph, threshold, cost, parities);
    }

    TSizeVec& offsets() {
        return this->maths::CBootstrapClusterer<POINT>::offsets();
    }
};

using TBootstrapClustererForTest2 = CBootstrapClustererForTest<TVector2>;
using TGraph = TBootstrapClustererForTest2::TGraph;
using TVertexItr = boost::graph_traits<TGraph>::vertex_iterator;
using TEdgeItr = boost::graph_traits<TGraph>::edge_iterator;
using TAdjacencyItr = boost::graph_traits<TGraph>::adjacency_iterator;

void clique(std::size_t a, std::size_t b, TGraph& graph) {
    for (std::size_t i = a; i < b; ++i) {
        for (std::size_t j = i + 1; j < b; ++j) {
            boost::put(boost::edge_weight, graph, boost::add_edge(i, j, graph).first, 1.0);
        }
    }
}

void connect(const TSizeVec& U, const TSizeVec& V, TGraph& graph) {
    BOOST_REQUIRE_EQUAL(U.size(), V.size());
    for (std::size_t i = 0; i < U.size(); ++i) {
        boost::put(boost::edge_weight, graph, boost::add_edge(U[i], V[i], graph).first, 1.0);
    }
}
}

BOOST_AUTO_TEST_CASE(testFacade) {
    // Check that clustering by facade produces the sample result.

    std::size_t improveParamsKmeansIterations = 4;
    std::size_t improveStructureClusterSeeds = 2;
    std::size_t improveStructureKmeansIterations = 3;

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "Trial " << t);

        double m1_[] = {2.0, 2.0};
        double v1_[] = {4.0, 2.0, 4.0};
        TVector2 m1(&m1_[0], &m1_[2]);
        TMatrix2 v1(&v1_[0], &v1_[3]);
        TVector2Vec points1;
        maths::CSampling::multivariateNormalSample(m1, v1, 50, points1);
        double m2_[] = {10.0, 5.0};
        double v2_[] = {4.0, 0.0, 1.0};
        TVector2 m2(&m2_[0], &m2_[2]);
        TMatrix2 v2(&v2_[0], &v2_[3]);
        TVector2Vec points2;
        maths::CSampling::multivariateNormalSample(m2, v2, 50, points2);
        TVector2Vec points;
        points.assign(points1.begin(), points1.end());
        points.insert(points.end(), points2.begin(), points2.end());
        std::sort(points.begin(), points.end());

        {
            maths::CXMeans<TVector2, maths::CGaussianInfoCriterion<TVector2, maths::E_BIC>> xmeans(20);

            maths::CSampling::seed();

            maths::CBootstrapClustererFacade<maths::CXMeans<TVector2, maths::CGaussianInfoCriterion<TVector2, maths::E_BIC>>> clusterer(
                xmeans, improveParamsKmeansIterations,
                improveStructureClusterSeeds, improveStructureKmeansIterations);

            TVector2VecVec actual;
            {
                TSizeVecVec clusters;
                clusterer.cluster(points, clusters);
                actual.resize(clusters.size());
                for (std::size_t i = 0; i < clusters.size(); ++i) {
                    std::sort(clusters[i].begin(), clusters[i].end());
                    for (std::size_t j = 0; j < clusters[i].size(); ++j) {
                        actual[i].push_back(points[clusters[i][j]]);
                    }
                }
            }

            maths::CSampling::seed();

            xmeans.setPoints(points);
            xmeans.run(improveParamsKmeansIterations, improveStructureClusterSeeds,
                       improveStructureKmeansIterations);

            TVector2VecVec expected(xmeans.clusters().size());
            for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
                expected[i] = xmeans.clusters()[i].points();
                std::sort(expected[i].begin(), expected[i].end());
            }

            BOOST_REQUIRE_EQUAL(expected.size(), actual.size());
            for (std::size_t i = 0; i < expected.size(); ++i) {
                BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expected[i]),
                                    core::CContainerPrinter::print(actual[i]));
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testBuildClusterGraph) {
    // Test we get the graph edges we expect for different overlap
    // thresholds.

    const std::size_t _ = 15;
    std::size_t clusters_[][5][5] = {
        {{0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}, {10, 11, 12, 13, 14}, {_, _, _, _, _}, {_, _, _, _, _}},
        {{0, 1, _, 3, 4}, {5, 6, _, _, _}, {10, 11, 12, 13, 14}, {2, 7, 8, 9, _}, {_, _, _, _, _}},
        {{0, 1, 2, 3, _}, {5, 6, 7, 8, 9}, {_, _, 12, 13, 14}, {4, _, _, _, _}, {10, 11, _, _, _}},
        {{_, _, 2, 3, 4}, {_, _, _, 8, 9}, {10, 11, 12, 13, 14}, {0, 1, 5, 6, 7}, {_, _, _, _, _}}};
    TBootstrapClustererForTest2::TSizeVecVecVec clusters(boost::size(clusters_));
    for (std::size_t i = 0; i < boost::size(clusters_); ++i) {
        for (std::size_t j = 0; j < boost::size(clusters_[i]); ++j) {
            TSizeVec cluster;
            for (std::size_t k = 0; k < boost::size(clusters_[i][j]); ++k) {
                if (clusters_[i][j][k] != _) {
                    cluster.push_back(clusters_[i][j][k]);
                }
            }
            if (!cluster.empty()) {
                clusters[i].push_back(cluster);
            }
        }
    }

    double overlaps[] = {0.1, 0.5, 0.9};
    std::string expected[] = {std::string("0: [3, 6, 7, 10, 12, 15]\n"
                                          "1: [4, 6, 8, 13, 15]\n"
                                          "2: [5, 9, 11, 14]\n"
                                          "3: [0, 7, 10, 12, 15]\n"
                                          "4: [1, 8, 15]\n"
                                          "5: [2, 9, 11, 14]\n"
                                          "6: [0, 1, 7, 8, 12, 13, 15]\n"
                                          "7: [0, 3, 6, 12, 15]\n"
                                          "8: [1, 4, 6, 13, 15]\n"
                                          "9: [2, 5, 14]\n"
                                          "10: [0, 3, 12]\n"
                                          "11: [2, 5, 14]\n"
                                          "12: [0, 3, 6, 7, 10]\n"
                                          "13: [1, 6, 8]\n"
                                          "14: [2, 5, 9, 11]\n"
                                          "15: [0, 1, 3, 4, 6, 7, 8]\n"),
                              std::string("0: [3, 7, 10, 12]\n"
                                          "1: [4, 6, 8, 13, 15]\n"
                                          "2: [5, 9, 11, 14]\n"
                                          "3: [0, 7, 10, 12]\n"
                                          "4: [1, 8, 15]\n"
                                          "5: [2, 9, 11, 14]\n"
                                          "6: [1, 8, 13]\n"
                                          "7: [0, 3, 12]\n"
                                          "8: [1, 4, 6, 13, 15]\n"
                                          "9: [2, 5, 14]\n"
                                          "10: [0, 3, 12]\n"
                                          "11: [2, 5, 14]\n"
                                          "12: [0, 3, 7, 10]\n"
                                          "13: [1, 6, 8]\n"
                                          "14: [2, 5, 9, 11]\n"
                                          "15: [1, 4, 8]\n"),
                              std::string("0: [3, 7, 10, 12]\n"
                                          "1: [4, 8, 13]\n"
                                          "2: [5, 9, 11, 14]\n"
                                          "3: [0, 10]\n"
                                          "4: [1, 8, 15]\n"
                                          "5: [2, 9, 11, 14]\n"
                                          "6: [13]\n"
                                          "7: [0]\n"
                                          "8: [1, 4, 13]\n"
                                          "9: [2, 5, 14]\n"
                                          "10: [0, 3, 12]\n"
                                          "11: [2, 5, 14]\n"
                                          "12: [0, 10]\n"
                                          "13: [1, 6, 8]\n"
                                          "14: [2, 5, 9, 11]\n"
                                          "15: [4]\n")};

    for (std::size_t i = 0; i < boost::size(overlaps); ++i) {
        LOG_DEBUG(<< "*** overlap threshold = " << overlaps[i] << " ***");

        TGraph graph;
        TBootstrapClustererForTest2 clusterer(overlaps[i], 1.0);
        clusterer.offsets().push_back(0);
        clusterer.offsets().push_back(3);
        clusterer.offsets().push_back(7);
        clusterer.offsets().push_back(12);
        clusterer.buildClusterGraph(clusters, graph);

        std::string rep;

        TVertexItr j, endj;
        for (boost::tie(j, endj) = boost::vertices(graph); j != endj; ++j) {
            rep += core::CStringUtils::typeToString(*j);
            TAdjacencyItr k, endk;
            boost::tie(k, endk) = boost::adjacent_vertices(*j, graph);
            TSizeVec adjacent(k, endk);
            std::sort(adjacent.begin(), adjacent.end());
            rep += ": " + core::CContainerPrinter::print(adjacent) + "\n";
        }

        LOG_DEBUG(<< "Overlap graph " << rep);
        BOOST_REQUIRE_EQUAL(expected[i], rep);
    }
}

BOOST_AUTO_TEST_CASE(testCutSearch) {
    // Test we generally find the sparsest cut in a graph with two cliques.

    std::size_t trials = 50;

    test::CRandomNumbers rng;

    TSizeVec splits;
    rng.generateUniformSamples(5, 15, trials, splits);
    TSizeVec connections;
    rng.generateUniformSamples(1, 15, trials, connections);

    TMeanAccumulator quality;
    for (std::size_t t = 0; t < trials; ++t) {
        std::size_t v = 20;

        TGraph graph(v);

        // Add the edges.
        std::size_t k = splits[t];
        clique(0, k, graph);
        clique(k, v, graph);
        TSizeVec U, V;
        rng.generateUniformSamples(0, k, connections[t], U);
        rng.generateUniformSamples(k, v, connections[t], V);
        connect(U, V, graph);

        LOG_DEBUG(<< "split = " << splits[t] << ":" << 20 - splits[t]);
        LOG_DEBUG(<< "# connections = " << connections[t]);

        TBootstrapClustererForTest2 clusterer(0.3, 3.0);

        double cost;
        TBoolVec parities;
        clusterer.cutSearch(0, 1, graph, 0.0, cost, parities);

        LOG_DEBUG(<< "cost = " << cost
                  << ", parities = " << core::CContainerPrinter::print(parities));

        double sparsestCut = static_cast<double>(connections[t]) /
                             static_cast<double>(20 - splits[t]) /
                             static_cast<double>(splits[t]);

        LOG_DEBUG(<< "sparsest = " << sparsestCut);
        quality.add(cost - sparsestCut);
    }

    LOG_DEBUG(<< "quality = " << 1.0 - maths::CBasicStatistics::mean(quality));
    BOOST_TEST_REQUIRE(1.0 - maths::CBasicStatistics::mean(quality) > 0.98);
}

BOOST_AUTO_TEST_CASE(testSeparate) {
    // Test we separate a graph with three cliques when we can.

    test::CRandomNumbers rng;

    const std::size_t trials = 100;

    TSizeVec splits1;
    rng.generateUniformSamples(5, 15, trials, splits1);
    TSizeVec splits2;
    rng.generateUniformSamples(25, 35, trials, splits2);
    TSizeVec connections;
    rng.generateUniformSamples(1, 15, 2 * trials, connections);

    std::size_t errors = 0;
    TMeanAccumulator quality;
    for (std::size_t t = 0; t < trials; ++t) {
        std::size_t v = 40;

        TGraph graph(v);

        std::size_t k[] = {splits1[t], splits2[t]};
        clique(0, k[0], graph);
        clique(k[0], k[1], graph);
        clique(k[1], v, graph);

        TSizeVec S, T, U, V;
        rng.generateUniformSamples(0, k[0], connections[2 * t], S);
        rng.generateUniformSamples(k[0], k[1], connections[2 * t], T);
        rng.generateUniformSamples(k[0], k[1], connections[2 * t + 1], U);
        rng.generateUniformSamples(k[1], v, connections[2 * t + 1], V);
        connect(S, T, graph);
        connect(U, V, graph);

        std::size_t e = boost::num_edges(graph);

        LOG_DEBUG(<< "split = " << splits1[t] << ":" << splits2[t] << ":"
                  << v - splits2[t]);
        LOG_DEBUG(<< "# connections = " << connections[2 * t] << " "
                  << connections[2 * t + 1]);

        TBootstrapClustererForTest2 clusterer(0.3, 3.0);

        TBoolVec parities;
        bool separable = clusterer.separate(graph, parities);
        LOG_DEBUG(<< "parities = " << core::CContainerPrinter::print(parities));

        double a = 0.0;
        double b = 0.0;
        double cut = 0.0;
        for (std::size_t i = 0; i < v; ++i) {
            (parities[i] ? a : b) += 1.0;
        }
        TEdgeItr i, end;
        for (boost::tie(i, end) = boost::edges(graph); i != end; ++i) {
            if (parities[boost::source(*i, graph)] != parities[boost::target(*i, graph)]) {
                cut += 1.0;
            }
        }
        LOG_DEBUG(<< "cost = " << cut / (a * b));

        double sparsestCut = std::min(
            static_cast<double>(connections[2 * t]) /
                static_cast<double>(k[0]) / static_cast<double>(v - k[0]),
            static_cast<double>(connections[2 * t + 1]) /
                static_cast<double>(k[1]) / static_cast<double>(v - k[1]));

        double threshold = 0.1 * static_cast<double>(2 * e) /
                           static_cast<double>(v * (v - 1));

        LOG_DEBUG(<< "sparsest = " << sparsestCut << " need " << threshold << " to separate");

        errors += static_cast<std::size_t>((sparsestCut < threshold) != separable);
        quality.add(cut / (a * b) - sparsestCut);
    }

    LOG_DEBUG(<< "errors = " << errors);
    LOG_DEBUG(<< "quality = " << 1.0 - maths::CBasicStatistics::mean(quality));
    BOOST_TEST_REQUIRE(errors < 4);
    BOOST_TEST_REQUIRE(1.0 - maths::CBasicStatistics::mean(quality) > 0.99);
}

BOOST_AUTO_TEST_CASE(testThickets) {
    // Test we find the correct thickets in a graph with two
    // components and three cliques.

    test::CRandomNumbers rng;

    const std::size_t trials = 10;

    TSizeVec splits1;
    rng.generateUniformSamples(5, 15, trials, splits1);
    TSizeVec splits2;
    rng.generateUniformSamples(25, 35, trials, splits2);
    TSizeVec connections;
    rng.generateUniformSamples(1, 10, trials, connections);

    int error = 0;

    TMeanAccumulator meanJaccard;
    for (std::size_t t = 0; t < trials; ++t) {
        std::size_t v = 40;

        TGraph graph(v);

        std::size_t k[] = {splits1[t], splits2[t]};
        clique(0, k[0], graph);
        clique(k[0], k[1], graph);
        clique(k[1], v, graph);

        TSizeVecVec expectedClusters(3);
        for (std::size_t i = 0; i < v; ++i) {
            if (i < k[0]) {
                expectedClusters[0].push_back(i);
            } else if (i < k[1]) {
                expectedClusters[1].push_back(i);
            } else {
                expectedClusters[2].push_back(i);
            }
        }
        std::sort(expectedClusters.begin(), expectedClusters.end());

        TSizeVec U, V;
        rng.generateUniformSamples(0, k[0], connections[t], U);
        rng.generateUniformSamples(k[0], k[1], connections[t], V);
        connect(U, V, graph);

        LOG_DEBUG(<< "split = " << splits1[t] << ":" << splits2[t] << ":"
                  << v - splits2[t]);
        LOG_DEBUG(<< "# connections = " << connections[t]);

        TSizeVec components(v);
        std::size_t c = boost::connected_components(graph, &components[0]);

        TBootstrapClustererForTest2 clusterer(0.3, 3.0);

        c = clusterer.thickets(c, graph, components);
        LOG_DEBUG(<< "components = " << core::CContainerPrinter::print(components));

        error += std::abs(3 - static_cast<int>(c));
        if (c == 3) {
            TSizeVecVec clusters(3);
            for (std::size_t i = 0; i < v; ++i) {
                clusters[components[i]].push_back(i);
            }
            std::sort(clusters.begin(), clusters.end());

            for (std::size_t i = 0; i < 3; ++i) {
                double jaccard = maths::CSetTools::jaccard(
                    expectedClusters[i].begin(), expectedClusters[i].end(),
                    clusters[i].begin(), clusters[i].end());
                BOOST_TEST_REQUIRE(jaccard > 0.8);
                meanJaccard.add(jaccard);
            }
        }
    }

    LOG_DEBUG(<< "error = " << error);
    LOG_DEBUG(<< "mean Jaccard = " << maths::CBasicStatistics::mean(meanJaccard));
    BOOST_TEST_REQUIRE(error < 2);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanJaccard) > 0.99);
}

BOOST_AUTO_TEST_CASE(testNonConvexClustering) {
    // Check the improvement in clustering when the underlying
    // assumptions of x-means (specifically cluster convexness
    // and Gaussian noise) are violated.

    test::CRandomNumbers rng;

    // We'd ideally find three clusters in test data. One of them
    // has mean equal to half a sine wave which poses problems for
    // x-means.

    double x[][2] = {
        {2.00000, 1.99667}, // Cluster 1
        {4.00000, 3.97339},    {6.00000, 5.91040},    {8.00000, 7.78837},
        {10.00000, 9.58851},   {12.00000, 11.29285},  {14.00000, 12.88435},
        {16.00000, 14.34712},  {18.00000, 15.66654},  {20.00000, 16.82942},
        {22.00000, 17.82415},  {24.00000, 18.64078},  {26.00000, 19.27116},
        {28.00000, 19.70899},  {30.00000, 19.94990},  {32.00000, 19.99147},
        {34.00000, 19.83330},  {36.00000, 19.47695},  {38.00000, 18.92600},
        {40.00000, 18.18595},  {42.00000, 17.26419},  {44.00000, 16.16993},
        {46.00000, 14.91410},  {48.00000, 13.50926},  {50.00000, 11.96944},
        {52.00000, 10.31003},  {54.00000, 8.54760},   {56.00000, 6.69976},
        {58.00000, 4.78499},   {60.00000, 2.82240},   {62.00000, 0.83161},
        {181.00000, 9.95004}, // Cluster 2
        {182.00000, 9.80067},  {183.00000, 9.55336},  {184.00000, 9.21061},
        {185.00000, 8.77583},  {186.00000, 8.25336},  {187.00000, 7.64842},
        {188.00000, 6.96707},  {189.00000, 6.21610},  {190.00000, 5.40302},
        {191.00000, 4.53596},  {192.00000, 3.62358},  {193.00000, 2.67499},
        {194.00000, 1.69967},  {195.00000, 0.70737},  {196.00000, -0.29200},
        {197.00000, -1.28844}, {198.00000, -2.27202}, {199.00000, -3.23290},
        {200.00000, -4.16147}, {201.00000, -5.04846}, {202.00000, -5.88501},
        {203.00000, -6.66276}, {204.00000, -7.37394}, {205.00000, -8.01144},
        {206.00000, -8.56889}, {207.00000, -9.04072}, {208.00000, -9.42222},
        {209.00000, -9.70958}, {210.00000, -9.89992}, {211.00000, -9.99135},
        {232.41593, -9.95004}, // Cluster 3
        {233.41593, -9.80067}, {234.41593, -9.55336}, {235.41593, -9.21061},
        {236.41593, -8.77583}, {237.41593, -8.25336}, {238.41593, -7.64842},
        {239.41593, -6.96707}, {240.41593, -6.21610}, {241.41593, -5.40302},
        {242.41593, -4.53596}, {243.41593, -3.62358}, {244.41593, -2.67499},
        {245.41593, -1.69967}, {246.41593, -0.70737}, {247.41593, 0.29200},
        {248.41593, 1.28844},  {249.41593, 2.27202},  {250.41593, 3.23290},
        {251.41593, 4.16147},  {252.41593, 5.04846},  {253.41593, 5.88501},
        {254.41593, 6.66276},  {255.41593, 7.37394},  {256.41593, 8.01144},
        {257.41593, 8.56889},  {258.41593, 9.04072},  {259.41593, 9.42222},
        {260.41593, 9.70958},  {261.41593, 9.89992},  {262.41593, 9.99135}};
    std::size_t clusters[] = {0, 31, 62, boost::size(x)};

    TSizeVecVec perfect(3);
    for (std::size_t i = 1; i < boost::size(clusters); ++i) {
        for (std::size_t j = clusters[i - 1]; j < clusters[i]; ++j) {
            perfect[i - 1].push_back(j);
        }
    }
    TSizeVecVec bootstrap;
    TSizeVecVec vanilla;

    TMeanAccumulator jaccardBootstrapToPerfect;
    TMeanAccumulator numberClustersBootstrap;
    TMeanAccumulator jaccardVanillaToPerfect;
    TMeanAccumulator numberClustersVanilla;

    std::size_t improveParamsKmeansIterations = 4;
    std::size_t improveStructureClusterSeeds = 2;
    std::size_t improveStructureKmeansIterations = 3;

    TVector2Vec flatPoints;
    TVector2SizeUMap lookup;
    TDoubleVec noise;
    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "Trial " << t);

        flatPoints.clear();
        lookup.clear();
        rng.generateUniformSamples(0, 4.0, 2 * boost::size(x), noise);
        for (std::size_t i = 0; i < boost::size(x); ++i) {
            TVector2 point(&x[i][0], &x[i][2]);
            point(0) += noise[2 * i];
            point(1) += noise[2 * i + 1];
            flatPoints.push_back(point);
            lookup[point] = i;
        }
        maths::CXMeans<TVector2, maths::CGaussianInfoCriterion<TVector2, maths::E_BIC>> xmeans(20);

        TVector2VecVec bootstrapClusters;
        maths::bootstrapCluster(flatPoints,
                                20, // trials
                                xmeans, improveParamsKmeansIterations,
                                improveStructureClusterSeeds, improveStructureKmeansIterations,
                                0.3, // overlap threshold to connect
                                3.0, // the degree of connection between overlapping clusters
                                bootstrapClusters);

        bootstrap.resize(bootstrapClusters.size());
        for (std::size_t i = 0; i < bootstrapClusters.size(); ++i) {
            bootstrap[i].clear();
            for (std::size_t j = 0; j < bootstrapClusters[i].size(); ++j) {
                auto k = lookup.find(bootstrapClusters[i][j]);
                BOOST_TEST_REQUIRE(k != lookup.end());
                bootstrap[i].push_back(k->second);
            }
            std::sort(bootstrap[i].begin(), bootstrap[i].end());
        }
        TDoubleVec jaccard;
        for (std::size_t i = 0; i < perfect.size(); ++i) {
            double jmax = 0.0;
            for (std::size_t j = 0; j < bootstrap.size(); ++j) {
                jmax = std::max(jmax, maths::CSetTools::jaccard(
                                          bootstrap[j].begin(), bootstrap[j].end(),
                                          perfect[i].begin(), perfect[i].end()));
            }
            jaccard.push_back(jmax);
        }
        LOG_DEBUG(<< "# clusters bootstrap = " << bootstrap.size()
                  << ", Jaccard bootstrap = " << core::CContainerPrinter::print(jaccard));
        numberClustersBootstrap.add(static_cast<double>(bootstrap.size()));
        jaccardBootstrapToPerfect.add(jaccard);

        TVector2Vec flatPoints_(flatPoints);
        xmeans.setPoints(flatPoints_);
        xmeans.run(improveParamsKmeansIterations, improveStructureClusterSeeds,
                   improveStructureKmeansIterations);

        vanilla.resize(xmeans.clusters().size());
        for (std::size_t i = 0; i < xmeans.clusters().size(); ++i) {
            vanilla[i].clear();
            for (std::size_t j = 0; j < xmeans.clusters()[i].points().size(); ++j) {
                auto k = lookup.find(xmeans.clusters()[i].points()[j]);
                BOOST_TEST_REQUIRE(k != lookup.end());
                vanilla[i].push_back(k->second);
            }
            std::sort(vanilla[i].begin(), vanilla[i].end());
        }
        jaccard.clear();
        for (std::size_t i = 0; i < perfect.size(); ++i) {
            double jmax = 0.0;
            for (std::size_t j = 0; j < vanilla.size(); ++j) {
                jmax = std::max(jmax, maths::CSetTools::jaccard(
                                          vanilla[j].begin(), vanilla[j].end(),
                                          perfect[i].begin(), perfect[i].end()));
            }
            jaccard.push_back(jmax);
        }
        LOG_DEBUG(<< "# clusters vanilla   = " << vanilla.size()
                  << ", Jaccard vanilla   = " << core::CContainerPrinter::print(jaccard));
        numberClustersVanilla.add(static_cast<double>(vanilla.size()));
        jaccardVanillaToPerfect.add(jaccard);
    }

    LOG_DEBUG(<< "Jaccard bootstrap to perfect = "
              << maths::CBasicStatistics::mean(jaccardBootstrapToPerfect));
    LOG_DEBUG(<< "Jaccard vanilla to perfect   = "
              << maths::CBasicStatistics::mean(jaccardVanillaToPerfect));
    LOG_DEBUG(<< "# clusters bootstrap = "
              << maths::CBasicStatistics::mean(numberClustersBootstrap));
    LOG_DEBUG(<< "# clusters vanilla   = "
              << maths::CBasicStatistics::mean(numberClustersVanilla));

    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        1.0, maths::CBasicStatistics::mean(jaccardBootstrapToPerfect), 0.1);
    BOOST_REQUIRE_CLOSE_ABSOLUTE(
        3.0, maths::CBasicStatistics::mean(numberClustersBootstrap), 0.6);
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(jaccardBootstrapToPerfect) >
                       maths::CBasicStatistics::mean(jaccardVanillaToPerfect));
}

BOOST_AUTO_TEST_CASE(testClusteringStability) {
    // Test that when we think there is sufficient certainty
    // to create clusters the assignment of points to clusters
    // is stable over multiple samplings of the data.

    test::CRandomNumbers rng;

    double m1_[] = {2.0, 2.0};
    double v1_[] = {4.0, 2.0, 4.0};
    TVector2 m1(&m1_[0], &m1_[2]);
    TMatrix2 v1(&v1_[0], &v1_[3]);
    TVector2Vec points1;
    maths::CSampling::multivariateNormalSample(m1, v1, 50, points1);
    double m2_[] = {10.0, 5.0};
    double v2_[] = {4.0, 0.0, 1.0};
    TVector2 m2(&m2_[0], &m2_[2]);
    TMatrix2 v2(&v2_[0], &v2_[3]);
    TVector2Vec points2;
    maths::CSampling::multivariateNormalSample(m2, v2, 50, points2);

    TSizeVecVec perfect(2);
    for (std::size_t i = 0; i < points1.size(); ++i) {
        perfect[0].push_back(i);
    }
    for (std::size_t i = 0; i < points2.size(); ++i) {
        perfect[1].push_back(points1.size() + i);
    }

    TSizeVecVec bootstrap;

    TVector2Vec points;
    points.insert(points.end(), points1.begin(), points1.end());
    points.insert(points.end(), points2.begin(), points2.end());

    TVector2SizeUMap lookup;
    for (std::size_t i = 0; i < points.size(); ++i) {
        lookup[points[i]] = i;
    }

    TSizeVecVec clusterCounts(perfect.size(), TSizeVec(points.size(), 0));

    for (std::size_t t = 0; t < 10; ++t) {
        LOG_DEBUG(<< "Trial " << t);

        rng.random_shuffle(points1.begin(), points1.end());
        rng.random_shuffle(points2.begin(), points2.end());

        points.assign(points1.begin(), points1.begin() + (3 * points1.size()) / 4);
        points.insert(points.end(), points2.begin(),
                      points2.begin() + (3 * points2.size()) / 4);

        TVector2VecVec bootstrapClusters;
        maths::CXMeans<TVector2, maths::CGaussianInfoCriterion<TVector2, maths::E_BIC>> xmeans(20);
        maths::bootstrapCluster(points,
                                20, // trials
                                xmeans,
                                4,   // improve params
                                2,   // improve structure seeds
                                3,   // improve structure
                                0.3, // overlap threshold to connect
                                3.0, // the degree of connection between overlapping clusters
                                bootstrapClusters);

        LOG_DEBUG(<< "# clusters = " << bootstrapClusters.size());
        if (bootstrapClusters.size() > 1) {
            bootstrap.resize(bootstrapClusters.size());
            for (std::size_t i = 0; i < bootstrapClusters.size(); ++i) {
                bootstrap[i].clear();
                for (std::size_t j = 0; j < bootstrapClusters[i].size(); ++j) {
                    auto k = lookup.find(bootstrapClusters[i][j]);
                    BOOST_TEST_REQUIRE(k != lookup.end());
                    bootstrap[i].push_back(k->second);
                }
                std::sort(bootstrap[i].begin(), bootstrap[i].end());
            }

            LOG_DEBUG(<< "clusters = " << core::CContainerPrinter::print(bootstrap));
            for (std::size_t i = 0; i < bootstrap.size(); ++i) {
                double Jmax = 0.0;
                std::size_t cluster = 0;
                for (std::size_t j = 0; j < perfect.size(); ++j) {
                    double J = maths::CSetTools::jaccard(
                        bootstrap[i].begin(), bootstrap[i].end(),
                        perfect[j].begin(), perfect[j].end());
                    boost::tie(Jmax, cluster) = std::max(
                        std::make_pair(Jmax, cluster), std::make_pair(J, j));
                }
                for (std::size_t j = 0; j < bootstrap[i].size(); ++j) {
                    ++clusterCounts[cluster][bootstrap[i][j]];
                }
            }
        }
    }

    TDoubleVec consistency(points.size(), 1.0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        double c0 = static_cast<double>(clusterCounts[0][i]);
        double c1 = static_cast<double>(clusterCounts[1][i]);
        if (c0 > 0.0 || c1 > 0.0) {
            consistency[i] = (std::max(c0, c1) - std::min(c0, c1)) / (c0 + c1);
        }
    }

    LOG_DEBUG(<< "consistency = " << core::CContainerPrinter::print(consistency));

    TMeanAccumulator meanConsistency;
    meanConsistency.add(consistency);
    LOG_DEBUG(<< "mean = " << maths::CBasicStatistics::mean(meanConsistency));
    BOOST_TEST_REQUIRE(maths::CBasicStatistics::mean(meanConsistency) > 0.95);
}

BOOST_AUTO_TEST_SUITE_END()
