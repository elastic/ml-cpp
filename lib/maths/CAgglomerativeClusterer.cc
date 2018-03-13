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

#include <maths/CAgglomerativeClusterer.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStringUtils.h>
#include <core/CoreTypes.h>

#include <maths/COrderings.h>
#include <maths/CSetTools.h>

#include <boost/numeric/conversion/bounds.hpp>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <math.h>

namespace ml {
namespace maths {

namespace {

typedef std::pair<std::size_t, std::size_t> TSizeSizePr;
typedef std::pair<double, TSizeSizePr> TDoubleSizeSizePrPr;
typedef std::vector<TDoubleSizeSizePrPr> TDoubleSizeSizePrPrVec;
typedef CAgglomerativeClusterer::TSizeVec TSizeVec;
typedef CAgglomerativeClusterer::TDoubleVec TDoubleVec;
typedef CAgglomerativeClusterer::TDoubleVecVec TDoubleVecVec;
typedef CAgglomerativeClusterer::CNode TNode;
typedef CAgglomerativeClusterer::TNodeVec TNodeVec;
typedef std::vector<TNode *> TNodePtrVec;

const double INF = boost::numeric::bounds<double>::highest();

//! Get the distance between node \p i and \p j.
inline double &distance(TDoubleVecVec &distanceMatrix, std::size_t i, std::size_t j) {
    if (j > i) {
        std::swap(i, j);
    }
    return distanceMatrix[i][j];
}

//! Get the distance between node \p i and \p j.
inline double distance(const TDoubleVecVec &distanceMatrix, std::size_t i, std::size_t j) {
    if (j > i) {
        std::swap(i, j);
    }
    return distanceMatrix[i][j];
}

//! \brief Complete update distance update function.
//!
//! The distance between clusters is given by
//! <pre class="fragment">
//!   \f$\displaystyle \max_{a \in A, b \in B}{d[a,b]}\f$
//! </pre>
struct SComplete {
    void operator()(const TDoubleVec & /*sizes*/,
                    std::size_t x,
                    std::size_t a,
                    std::size_t b,
                    TDoubleVecVec &distanceMatrix) const {
        distance(distanceMatrix, b, x) =
            std::max(distance(distanceMatrix, a, x), distance(distanceMatrix, b, x));
    }
};

//! \brief Average objective distance update function.
//!
//! The distance between clusters is given by
//! <pre class="fragment">
//!   \f$\displaystyle \frac{1}{|A||B|}\sum_{a \in A, b \in B}{d[a,b]}\f$
//! </pre>
struct SAverage {
    void operator()(const TDoubleVec &sizes,
                    std::size_t x,
                    std::size_t a,
                    std::size_t b,
                    TDoubleVecVec &distanceMatrix) const {
        double sa = sizes[a];
        double sb = sizes[b];
        distance(distanceMatrix, b, x) =
            (sa * distance(distanceMatrix, a, x) + sb * distance(distanceMatrix, b, x)) / (sa + sb);
    }
};

//! \brief Weighted objective distance update function.
struct SWeighted {
    void operator()(const TDoubleVec /*sizes*/,
                    std::size_t x,
                    std::size_t a,
                    std::size_t b,
                    TDoubleVecVec &distanceMatrix) const {
        distance(distanceMatrix, b, x) =
            (distance(distanceMatrix, a, x) + distance(distanceMatrix, b, x)) / 2.0;
    }
};

//! \brief Ward objective distance update function.
//!
//! See https://en.wikipedia.org/wiki/Ward%27s_method.
struct SWard {
    void operator()(const TDoubleVec sizes,
                    std::size_t x,
                    std::size_t a,
                    std::size_t b,
                    TDoubleVecVec &distanceMatrix) const {
        double sa = sizes[a];
        double sb = sizes[b];
        double sx = sizes[x];
        distance(distanceMatrix, b, x) = ::sqrt((sa + sx) * distance(distanceMatrix, a, x) +
                                                (sb + sx) * distance(distanceMatrix, b, x) -
                                                sx * distance(distanceMatrix, a, b)) /
                                         (sa + sb + sx);
    }
};

//! MST-LINKAGE algorithm due to Rolhf and given in Mullner.
//!
//! This lifts the subset of the minimum spanning tree
//! algorithm needed for building hierarchical clusterings.
//!
//! For details see http://arxiv.org/pdf/1109.2378.pdf.
//!
//! \param[in] distanceMatrix the matrices of distances
//! between the points to cluster.
//! \param[in] L Filled in with the unsorted dendrogram.
void mstCluster(const TDoubleVecVec &distanceMatrix, TDoubleSizeSizePrPrVec &L) {
    L.clear();
    std::size_t N = distanceMatrix.size();

    if (N <= 1) {
        return;
    }

    L.reserve(N - 1);

    TSizeVec S;
    S.reserve(N);
    for (std::size_t i = 0u; i < N; ++i) {
        S.push_back(i);
    }
    TDoubleVec D(N, INF);

    std::size_t c = S[N - 1];
    while (S.size() > 1) {
        S.erase(std::find(S.begin(), S.end(), c));

        std::size_t n = 0;
        double d = INF;

        for (std::size_t i = 0u; i < S.size(); ++i) {
            std::size_t x = S[i];
            D[x] = std::min(D[x], distance(distanceMatrix, x, c));
            if (D[x] < d) {
                n = x;
                d = D[x];
            }
        }

        L.emplace_back(d, std::make_pair(std::min(c, n), std::max(c, n)));
        c = n;
    }
}

//! The NN-CHAIN-LINKAGE cluster algorithm due to Murtagh
//! and given in Mullner.
//!
//! This makes use of the fact that reciprocal nearest
//! neighbours eventually get clustered together by some
//! node of the stepwise dendrogram for a certain class
//! of objective function.
//!
//! For details see http://arxiv.org/pdf/1109.2378.pdf.
//!
//! \param[in,out] distanceMatrix the matrices of distances
//! between the points to cluster.
//! \param[in] update The distance update function which varies
//! based on the objective function.
//! \param[in] L Filled in with the unsorted dendrogram.
//! \note This has worst case O(N^2) complexity.
//! \note For maximum efficiency modifications are made in
//! place to \p distanceMatrix.
template <typename UPDATE>
void nnCluster(TDoubleVecVec &distanceMatrix, UPDATE update, TDoubleSizeSizePrPrVec &L) {
    // In departure from the scheme given by Mullner we make all
    // our updates in-place by using a direct address table from
    // n -> max(a, b), where n is the new node index and a and b
    // are reciprocal nearest neighbours. It is still possible to
    // build the stepwise dendrogram from the resulting unsorted
    // dendrogram, we just need to keep track of the highest node
    // created for each index when building the tree. See buildTree
    // for details.

    L.clear();
    std::size_t N = distanceMatrix.size();

    if (N <= 1) {
        return;
    }

    L.reserve(N - 1);

    TSizeVec S;
    S.reserve(N);
    for (std::size_t i = 0u; i < N; ++i) {
        S.push_back(i);
    }
    TSizeVec chain;
    chain.reserve(N);
    TDoubleVec size(N, 1.0);
    TSizeVec rightmost;
    rightmost.reserve(2 * N - 1);
    for (std::size_t i = 0u; i < N; ++i) {
        rightmost.push_back(i);
    }

    std::size_t a = 0;
    std::size_t b = 1;
    std::size_t p = N - 1;

    while (S.size() > 1) {
        std::size_t m = chain.size();
        if (m <= 3) {
            a = S[0];
            b = S[1];
            chain.clear();
            chain.push_back(a);
            m = 1;
        } else {
            a = chain[m - 4];
            b = chain[m - 3];
            // Cut the tail.
            chain.pop_back();
            chain.pop_back();
            chain.pop_back();
            m -= 3;
        }

        LOG_TRACE("chain = " << core::CContainerPrinter::print(chain));
        LOG_TRACE("a = " << a << ", b = " << b << ", m = " << m);

        double d;
        do {
            std::size_t c = 0u;
            std::size_t ra = rightmost[a];
            d = INF;
            for (std::size_t i = 0u; i < S.size(); ++i) {
                std::size_t x = S[i];
                std::size_t rx = rightmost[x];
                if (a != x) {
                    double dx = distance(distanceMatrix, ra, rx);
                    if (dx < d || (dx == d && x == b)) {
                        c = x;
                        d = dx;
                    }
                }
            }
            b = a;
            a = c;
            chain.push_back(a);
            ++m;
        } while (m <= 3 || a != chain[m - 3]);

        if (a > b) {
            std::swap(a, b);
        }
        std::size_t ra = rightmost[a];
        std::size_t rb = rightmost[b];

        LOG_TRACE("chain = " << core::CContainerPrinter::print(chain));
        LOG_TRACE("d = " << d << ", a = " << a << ", b = " << b << ", rightmost a = " << ra
                         << ", rightmost b " << rb << ", m = " << m);

        // a and b are reciprocal nearest neighbors.
        L.emplace_back(d, std::make_pair(ra, rb));

        // Update the index set, the distance matrix, the sizes
        // and the rightmost direct address table.
        std::size_t merged[] = {a, b};
        CSetTools::inplace_set_difference(S, merged, merged + 2);
        for (std::size_t i = 0u; i < S.size(); ++i) {
            std::size_t x = S[i];
            update(size, rightmost[x], ra, rb, distanceMatrix);
        }
        size[rb] += size[ra];
        S.push_back(++p);
        rightmost[p] = rb;
    }
}

//! Add a node to the end of the tree with height \p height.
TNode &addNode(TNodeVec &tree, double height) {
    tree.emplace_back(tree.size(), height);
    return tree.back();
}

//! Build the binary hierarchical clustering tree from the
//! unsorted dendrogram representation in \p heights.
//!
//! \param[in,out] heights The nodes which are merged and
//! the level at which they are merged. This can contain
//! repeated node indices, in which case the later indices
//! refer to the last node created at that index. Note that
//! these are (stably) sorted.
//! \param[out] tree A binary tree representing the stepwise
//! dendrogram.
void buildTree(TDoubleSizeSizePrPrVec &heights, TNodeVec &tree) {
    tree.clear();

    std::size_t n = heights.size();

    if (n == 0) {
        return;
    }

    tree.reserve(2 * n + 1);
    for (std::size_t i = 0u; i <= n; ++i) {
        tree.emplace_back(i, 0.0);
    }

    std::stable_sort(heights.begin(), heights.end(), COrderings::SFirstLess());
    LOG_TRACE("heights = " << core::CContainerPrinter::print(heights));

    for (std::size_t i = 0u; i < n; ++i) {
        double h = heights[i].first;
        std::size_t j = heights[i].second.first;
        std::size_t k = heights[i].second.second;
        LOG_TRACE("Joining " << j << " and " << k << " at height " << h);
        TNode &parent = addNode(tree, h);
        parent.addChild(tree[j].root());
        parent.addChild(tree[k].root());
    }
}
}

bool CAgglomerativeClusterer::initialize(TDoubleVecVec &distanceMatrix) {
    // Check that the matrix is square.
    std::size_t n = distanceMatrix.size();
    for (std::size_t i = 0u; i < n; ++i) {
        LOG_TRACE("D = " << core::CContainerPrinter::print(distanceMatrix[i]));
        if (distanceMatrix[i].size() != i + 1) {
            LOG_ERROR("Distance matrix isn't upper triangular");
            return false;
        }
    }

    m_DistanceMatrix.swap(distanceMatrix);

    m_Pi.resize(n);
    for (std::size_t i = 0u; i < n; ++i) {
        m_Pi[i] = i;
    }
    m_Lambda.resize(n, INF);
    m_M.resize(n);

    return true;
}

void CAgglomerativeClusterer::run(EObjective objective, TNodeVec &tree) {
    if (m_DistanceMatrix.empty()) {
        return;
    }

    TDoubleSizeSizePrPrVec heights;

    switch (objective) {
        case E_Single:
            mstCluster(m_DistanceMatrix, heights);
            break;
        case E_Complete:
            nnCluster(m_DistanceMatrix, SComplete(), heights);
            break;
        case E_Average:
            nnCluster(m_DistanceMatrix, SAverage(), heights);
            break;
        case E_Weighted:
            nnCluster(m_DistanceMatrix, SWeighted(), heights);
            break;
        case E_Ward:
            nnCluster(m_DistanceMatrix, SWard(), heights);
            break;
    }

    buildTree(heights, tree);
}

////// CNode //////

CAgglomerativeClusterer::CNode::CNode(std::size_t index, double height)
    : m_Parent(0), m_LeftChild(0), m_RightChild(0), m_Index(index), m_Height(height) {}

bool CAgglomerativeClusterer::CNode::addChild(CNode &child) {
    if (!m_LeftChild) {
        m_LeftChild = &child;
        child.m_Parent = this;
        return true;
    }
    if (!m_RightChild) {
        m_RightChild = &child;
        child.m_Parent = this;
        return true;
    }

    LOG_ERROR("Trying to add third child");

    return false;
}

std::size_t CAgglomerativeClusterer::CNode::index(void) const { return m_Index; }

double CAgglomerativeClusterer::CNode::height(void) const { return m_Height; }

TNode &CAgglomerativeClusterer::CNode::root(void) {
    CNode *result = this;
    for (CNode *parent = m_Parent; parent; parent = parent->m_Parent) {
        result = parent;
    }
    return *result;
}

void CAgglomerativeClusterer::CNode::points(TSizeVec &result) const {
    if (!m_LeftChild && !m_RightChild) {
        result.push_back(m_Index);
    }
    if (m_LeftChild) {
        m_LeftChild->points(result);
    }
    if (m_RightChild) {
        m_RightChild->points(result);
    }
}

void CAgglomerativeClusterer::CNode::clusters(TDoubleSizeVecPrVec &result) const {
    if (m_LeftChild && m_RightChild) {
        TSizeVec points;
        this->points(points);
        result.emplace_back(m_Height, points);
    }
    if (m_LeftChild) {
        m_LeftChild->clusters(result);
    }
    if (m_RightChild) {
        m_RightChild->clusters(result);
    }
}

void CAgglomerativeClusterer::CNode::clusteringAt(double height, TSizeVecVec &result) const {
    if (height >= m_Height) {
        result.push_back(TSizeVec());
        this->points(result.back());
    } else {
        if (m_LeftChild && height < m_LeftChild->height()) {
            m_LeftChild->clusteringAt(height, result);
        } else if (m_LeftChild) {
            result.push_back(TSizeVec());
            m_LeftChild->points(result.back());
        }
        if (m_RightChild && height < m_RightChild->height()) {
            m_RightChild->clusteringAt(height, result);
        } else if (m_RightChild) {
            result.push_back(TSizeVec());
            m_RightChild->points(result.back());
        }
    }
}

std::string CAgglomerativeClusterer::CNode::print(const std::string &indent) const {
    std::string result;
    result += "height = " + core::CStringUtils::typeToStringPretty(m_Height);
    if (m_LeftChild) {
        result += core_t::LINE_ENDING + indent + m_LeftChild->print(indent + "  ");
    }
    if (m_RightChild) {
        result += core_t::LINE_ENDING + indent + m_RightChild->print(indent + "  ");
    }
    if (!m_LeftChild && !m_RightChild) {
        result += ", point = " + core::CStringUtils::typeToStringPretty(m_Index);
    }
    return result;
}
}
}
