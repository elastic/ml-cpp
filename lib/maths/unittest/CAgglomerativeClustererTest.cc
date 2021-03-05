/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CAgglomerativeClusterer.h>
#include <maths/COrderings.h>

#include <test/CRandomNumbers.h>

#include <boost/range.hpp>
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

BOOST_AUTO_TEST_SUITE(CAgglomerativeClustererTest)

using namespace ml;

namespace {

using TDoubleVec = std::vector<double>;
using TDoubleVecVec = std::vector<TDoubleVec>;
using TSizeVec = std::vector<std::size_t>;
using TSizeVecVec = std::vector<TSizeVec>;
using TDoubleSizeVecPr = std::pair<double, TSizeVec>;
using TDoubleSizeVecPrVec = std::vector<TDoubleSizeVecPr>;

class CCluster {
public:
    explicit CCluster(std::size_t p) : m_Height(0.0), m_Points(1, p) {}

    void swap(CCluster& other) { m_Points.swap(other.m_Points); }

    static CCluster merge(double height, const CCluster& lhs, const CCluster& rhs) {
        CCluster result;
        result.m_Height = height;
        result.m_Points.reserve(lhs.m_Points.size() + rhs.m_Points.size());
        result.m_Points.insert(result.m_Points.end(), lhs.m_Points.begin(),
                               lhs.m_Points.end());
        result.m_Points.insert(result.m_Points.end(), rhs.m_Points.begin(),
                               rhs.m_Points.end());
        std::sort(result.m_Points.begin(), result.m_Points.end());
        return result;
    }

    void add(TDoubleSizeVecPrVec& result) {
        result.push_back(TDoubleSizeVecPr(m_Height, m_Points));
    }

    const TSizeVec& points() const { return m_Points; }

private:
    explicit CCluster() : m_Height(0.0) {}

private:
    double m_Height;
    TSizeVec m_Points;
};

using TClusterVec = std::vector<CCluster>;

class CSlinkObjective {
public:
    CSlinkObjective(const TDoubleVecVec& distanceMatrix)
        : m_DistanceMatrix(&distanceMatrix) {}

    double operator()(const CCluster& lhs, const CCluster& rhs) {
        double result = std::numeric_limits<double>::max();
        const TSizeVec& lp = lhs.points();
        const TSizeVec& rp = rhs.points();
        for (std::size_t i = 0; i < lp.size(); ++i) {
            for (std::size_t j = 0; j < rp.size(); ++j) {
                std::size_t pi = lp[i];
                std::size_t pj = rp[j];
                if (pj > pi) {
                    std::swap(pi, pj);
                }
                result = std::min(result, (*m_DistanceMatrix)[pi][pj]);
            }
        }
        return result;
    }

private:
    const TDoubleVecVec* m_DistanceMatrix;
};

class CClinkObjective {
public:
    CClinkObjective(const TDoubleVecVec& distanceMatrix)
        : m_DistanceMatrix(&distanceMatrix) {}

    double operator()(const CCluster& lhs, const CCluster& rhs) {
        double result = -std::numeric_limits<double>::max();
        const TSizeVec& lp = lhs.points();
        const TSizeVec& rp = rhs.points();
        for (std::size_t i = 0; i < lp.size(); ++i) {
            for (std::size_t j = 0; j < rp.size(); ++j) {
                std::size_t pi = lp[i];
                std::size_t pj = rp[j];
                if (pj > pi) {
                    std::swap(pi, pj);
                }
                result = std::max(result, (*m_DistanceMatrix)[pi][pj]);
            }
        }
        return result;
    }

private:
    const TDoubleVecVec* m_DistanceMatrix;
};

template<typename OBJECTIVE>
TClusterVec agglomerativeCluster(const TDoubleVecVec& distanceMatrix) {
    std::size_t n = distanceMatrix.size();

    TClusterVec clusters;
    clusters.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        clusters.push_back(CCluster(i));
    }

    OBJECTIVE f(distanceMatrix);

    TClusterVec tree;
    tree.reserve(n);

    while (clusters.size() > 1) {
        double fmin = std::numeric_limits<double>::max();
        std::size_t mi = 0;
        std::size_t mj = 0;

        for (std::size_t i = 0; i < clusters.size(); ++i) {
            for (std::size_t j = i + 1; j < clusters.size(); ++j) {
                double fij = f(clusters[i], clusters[j]);
                if (fij < fmin) {
                    fmin = fij;
                    mi = i;
                    mj = j;
                }
            }
        }

        if (mi < mj) {
            std::swap(mi, mj);
        }
        LOG_DEBUG(<< "fmin = " << fmin << ", mi = " << mi << ", mj = " << mj);

        CCluster merged = CCluster::merge(fmin, clusters[mi], clusters[mj]);
        tree.push_back(merged);

        clusters.erase(clusters.begin() + mi);
        clusters.erase(clusters.begin() + mj);
        clusters.push_back(merged);
    }

    return tree;
}

std::string print(maths::CAgglomerativeClusterer::EObjective o) {
    switch (o) {
    case maths::CAgglomerativeClusterer::E_Single:
        return "slink";
    case maths::CAgglomerativeClusterer::E_Complete:
        return "clink";
    case maths::CAgglomerativeClusterer::E_Average:
        return "average";
    case maths::CAgglomerativeClusterer::E_Weighted:
        return "weighted";
    case maths::CAgglomerativeClusterer::E_Ward:
        return "ward";
    }
    return "unexpected";
}
}

BOOST_AUTO_TEST_CASE(testNode) {
    double heights[] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.5, 1.9, 4.0};

    maths::CAgglomerativeClusterer::CNode nodes[] = {
        maths::CAgglomerativeClusterer::CNode(0, heights[0]),
        maths::CAgglomerativeClusterer::CNode(1, heights[1]),
        maths::CAgglomerativeClusterer::CNode(2, heights[2]),
        maths::CAgglomerativeClusterer::CNode(3, heights[3]),
        maths::CAgglomerativeClusterer::CNode(4, heights[4]),
        maths::CAgglomerativeClusterer::CNode(5, heights[5]),
        maths::CAgglomerativeClusterer::CNode(6, heights[6]),
        maths::CAgglomerativeClusterer::CNode(7, heights[7]),
        maths::CAgglomerativeClusterer::CNode(8, heights[8])};

    nodes[5].addChild(nodes[0]);
    nodes[5].addChild(nodes[1]);
    nodes[6].addChild(nodes[2]);
    nodes[6].addChild(nodes[3]);
    nodes[7].addChild(nodes[4]);
    nodes[7].addChild(nodes[6]);
    nodes[8].addChild(nodes[5]);
    nodes[8].addChild(nodes[7]);

    const maths::CAgglomerativeClusterer::CNode& root = nodes[8];

    LOG_DEBUG(<< "tree = \n" << root.print());

    for (std::size_t i = 0; i < 8; ++i) {
        BOOST_REQUIRE_EQUAL(root.index(), nodes[i].root().index());
    }

    TSizeVec points;
    root.points(points);
    std::sort(points.begin(), points.end());
    BOOST_REQUIRE_EQUAL(std::string("[0, 1, 2, 3, 4]"),
                        core::CContainerPrinter::print(points));

    points.clear();
    nodes[7].points(points);
    std::sort(points.begin(), points.end());
    BOOST_REQUIRE_EQUAL(std::string("[2, 3, 4]"), core::CContainerPrinter::print(points));

    {
        TDoubleSizeVecPrVec clusters;
        root.clusters(clusters);
        std::sort(clusters.begin(), clusters.end(), maths::COrderings::SFirstLess());
        BOOST_REQUIRE_EQUAL(std::string("[(1, [0, 1]), (1.5, [2, 3]), (1.9, [4, 2, 3]), (4, [0, 1, 4, 2, 3])]"),
                            core::CContainerPrinter::print(clusters));
    }

    std::string expected[] = {std::string("[[0, 1], [2], [3], [4]]"),
                              std::string("[[0, 1], [2, 3], [4]]"),
                              std::string("[[0, 1], [4, 2, 3]]"),
                              std::string("[[0, 1, 4, 2, 3]]")};
    for (std::size_t h = 5; h < 9; ++h) {
        TSizeVecVec clusters;
        root.clusteringAt(heights[h], clusters);
        std::sort(clusters.begin(), clusters.end());
        LOG_DEBUG(<< "Clusters at " << heights[h] << " are "
                  << core::CContainerPrinter::print(clusters));
        BOOST_REQUIRE_EQUAL(expected[h - 5], core::CContainerPrinter::print(clusters));
    }
}

BOOST_AUTO_TEST_CASE(testSimplePermutations) {
    double x[] = {1.0, 3.2, 4.5, 7.8};
    std::size_t n = boost::size(x);

    maths::CAgglomerativeClusterer::EObjective objectives[] = {
        maths::CAgglomerativeClusterer::E_Single, maths::CAgglomerativeClusterer::E_Complete};

    std::string expected[] = {
        std::string("[(3.3, [0, 1, 2, 3]), (2.2, [0, 1, 2]), (1.3, [1, 2])]"),
        std::string("[(6.8, [0, 1, 2, 3]), (3.5, [0, 1, 2]), (1.3, [1, 2])]")};

    for (std::size_t o = 0; o < boost::size(objectives); ++o) {
        LOG_DEBUG(<< "****** " << print(objectives[o]) << " ******");

        std::size_t p[] = {0, 1, 2, 3};

        do {
            LOG_DEBUG(<< "*** " << core::CContainerPrinter::print(p) << " ***");

            TDoubleVecVec distanceMatrix(n);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = i; j < n; ++j) {
                    distanceMatrix[j].push_back(std::fabs(x[p[i]] - x[p[j]]));
                }
                LOG_DEBUG(<< "D = " << core::CContainerPrinter::print(distanceMatrix[i]));
            }

            maths::CAgglomerativeClusterer clusterer;
            BOOST_TEST_REQUIRE(clusterer.initialize(distanceMatrix));

            maths::CAgglomerativeClusterer::TNodeVec tree;
            clusterer.run(objectives[o], tree);

            TDoubleSizeVecPrVec clusters;
            tree.back().clusters(clusters);

            LOG_DEBUG(<< "clusters           = " << core::CContainerPrinter::print(clusters));

            for (std::size_t i = 0; i < clusters.size(); ++i) {
                for (std::size_t j = 0; j < clusters[i].second.size(); ++j) {
                    clusters[i].second[j] = p[clusters[i].second[j]];
                }
                std::sort(clusters[i].second.begin(), clusters[i].second.end());
            }

            LOG_DEBUG(<< "canonical clusters = " << core::CContainerPrinter::print(clusters));

            BOOST_REQUIRE_EQUAL(expected[o], core::CContainerPrinter::print(clusters));
        } while (std::next_permutation(std::begin(p), std::end(p)));
    }
}

BOOST_AUTO_TEST_CASE(testDegenerate) {
    double x[] = {1.0, 3.2, 3.2, 3.2, 4.5, 7.8};
    std::size_t n = boost::size(x);

    maths::CAgglomerativeClusterer::EObjective objectives[] = {
        maths::CAgglomerativeClusterer::E_Single, maths::CAgglomerativeClusterer::E_Complete};

    std::string expected[][3] = {
        {std::string("[(3.3, [0, 1, 2, 3, 4, 5]), (2.2, [0, 1, 2, 3, 4]), (1.3, [1, 2, 3, 4]), (0, [1, 2, 3]), (0, [1, 2])]"),
         std::string("[(3.3, [0, 1, 2, 3, 4, 5]), (2.2, [0, 1, 2, 3, 4]), (1.3, [1, 2, 3, 4]), (0, [1, 2, 3]), (0, [1, 3])]"),
         std::string("[(3.3, [0, 1, 2, 3, 4, 5]), (2.2, [0, 1, 2, 3, 4]), (1.3, [1, 2, 3, 4]), (0, [1, 2, 3]), (0, [2, 3])]")},
        {std::string("[(6.8, [0, 1, 2, 3, 4, 5]), (3.5, [0, 1, 2, 3, 4]), (1.3, [1, 2, 3, 4]), (0, [1, 2, 3]), (0, [1, 2])]"),
         std::string("[(6.8, [0, 1, 2, 3, 4, 5]), (3.5, [0, 1, 2, 3, 4]), (1.3, [1, 2, 3, 4]), (0, [1, 2, 3]), (0, [1, 3])]"),
         std::string("[(6.8, [0, 1, 2, 3, 4, 5]), (3.5, [0, 1, 2, 3, 4]), (1.3, [1, 2, 3, 4]), (0, [1, 2, 3]), (0, [2, 3])]")}};

    for (std::size_t o = 0u, count = 0; o < boost::size(objectives); ++o) {
        LOG_DEBUG(<< "****** " << print(objectives[o]) << " ******");

        std::size_t p[] = {0, 1, 2, 3, 4, 5};

        do {
            if (count % 10 == 0) {
                LOG_DEBUG(<< "*** " << core::CContainerPrinter::print(p) << " ***");
            }

            TDoubleVecVec distanceMatrix(n);
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = i; j < n; ++j) {
                    distanceMatrix[j].push_back(std::fabs(x[p[i]] - x[p[j]]));
                }
                if (count % 10 == 0) {
                    LOG_DEBUG(<< "D = "
                              << core::CContainerPrinter::print(distanceMatrix[i]));
                }
            }

            maths::CAgglomerativeClusterer clusterer;
            BOOST_TEST_REQUIRE(clusterer.initialize(distanceMatrix));

            maths::CAgglomerativeClusterer::TNodeVec tree;
            clusterer.run(objectives[o], tree);

            TDoubleSizeVecPrVec clusters;
            tree.back().clusters(clusters);

            if (count % 10 == 0) {
                LOG_DEBUG(<< "clusters           = "
                          << core::CContainerPrinter::print(clusters));
            }

            for (std::size_t i = 0; i < clusters.size(); ++i) {
                for (std::size_t j = 0; j < clusters[i].second.size(); ++j) {
                    clusters[i].second[j] = p[clusters[i].second[j]];
                }
                std::sort(clusters[i].second.begin(), clusters[i].second.end());
            }

            if (count % 10 == 0) {
                LOG_DEBUG(<< "canonical clusters = "
                          << core::CContainerPrinter::print(clusters));
            }

            BOOST_TEST_REQUIRE(
                (expected[o][0] == core::CContainerPrinter::print(clusters) ||
                 expected[o][1] == core::CContainerPrinter::print(clusters) ||
                 expected[o][2] == core::CContainerPrinter::print(clusters)));
            ++count;
        } while (std::next_permutation(std::begin(p), std::end(p)));
    }
}

BOOST_AUTO_TEST_CASE(testRandom) {
    test::CRandomNumbers rng;

    std::size_t n = 20;

    maths::CAgglomerativeClusterer::EObjective objectives[] = {
        maths::CAgglomerativeClusterer::E_Single, maths::CAgglomerativeClusterer::E_Complete};

    for (std::size_t o = 0; o < boost::size(objectives); ++o) {
        LOG_DEBUG(<< "*** " << print(objectives[o]) << " ***");

        for (std::size_t t = 0; t < 10; ++t) {
            TDoubleVec dij;
            rng.generateUniformSamples(0.0, 100.0, n * (n - 1) / 2, dij);

            TDoubleVecVec distanceMatrix(n);
            for (std::size_t i = 0u, k = 0; i < n; ++i) {
                for (std::size_t j = i; j < n; ++j) {
                    distanceMatrix[j].push_back(i == j ? 0.0 : dij[k++]);
                }
                LOG_DEBUG(<< "D = " << core::CContainerPrinter::print(distanceMatrix[i]));
            }

            TClusterVec expectedTree;
            switch (objectives[o]) {
            case maths::CAgglomerativeClusterer::E_Single:
                expectedTree = agglomerativeCluster<CSlinkObjective>(distanceMatrix);
                break;
            case maths::CAgglomerativeClusterer::E_Complete:
                expectedTree = agglomerativeCluster<CClinkObjective>(distanceMatrix);
                break;
            case maths::CAgglomerativeClusterer::E_Average:
            case maths::CAgglomerativeClusterer::E_Weighted:
            case maths::CAgglomerativeClusterer::E_Ward:
                // TODO
                BOOST_TEST_REQUIRE(false);
                break;
            }

            TDoubleSizeVecPrVec expectedClusters;
            expectedClusters.reserve(expectedTree.size());
            for (std::size_t i = 0; i < expectedTree.size(); ++i) {
                expectedTree[i].add(expectedClusters);
            }
            std::sort(expectedClusters.begin(), expectedClusters.end());

            LOG_DEBUG(<< "expected clusters = "
                      << core::CContainerPrinter::print(expectedClusters));

            maths::CAgglomerativeClusterer clusterer;
            BOOST_TEST_REQUIRE(clusterer.initialize(distanceMatrix));

            maths::CAgglomerativeClusterer::TNodeVec tree;
            clusterer.run(objectives[o], tree);

            TDoubleSizeVecPrVec clusters;
            tree.back().clusters(clusters);
            for (std::size_t i = 0; i < clusters.size(); ++i) {
                std::sort(clusters[i].second.begin(), clusters[i].second.end());
            }
            std::sort(clusters.begin(), clusters.end());

            LOG_DEBUG(<< "clusters          = " << core::CContainerPrinter::print(clusters));

            BOOST_REQUIRE_EQUAL(core::CContainerPrinter::print(expectedClusters),
                                core::CContainerPrinter::print(clusters));
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
