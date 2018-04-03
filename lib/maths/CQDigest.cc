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

#include <maths/CQDigest.h>

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>
#include <core/RestoreMacros.h>

#include <maths/CChecksum.h>

#include <boost/bind.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/tuple/tuple.hpp>

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <sstream>

#include <math.h>

namespace ml {
namespace maths {

namespace {
std::string EMPTY_STRING;
}

const std::string CQDigest::K_TAG("a");
const std::string CQDigest::N_TAG("b");
const std::string CQDigest::NODE_TAG("c");

CQDigest::CQDigest(uint64_t k, double decayRate)
    : m_K(k), m_N(0u), m_Root(0), m_NodeAllocator(static_cast<std::size_t>(3 * m_K + 2)), m_DecayRate(decayRate) {
    m_Root = &m_NodeAllocator.create(CNode(0, 1, 0, 0));
}

void CQDigest::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(K_TAG, m_K);
    inserter.insertValue(N_TAG, m_N);

    // Note the tree is serialized flat in pre-order.
    m_Root->persistRecursive(NODE_TAG, inserter);
}

bool CQDigest::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    std::size_t nodeCount = 0u;

    do {
        const std::string& name = traverser.name();
        RESTORE_BUILT_IN(K_TAG, m_K)
        RESTORE_BUILT_IN(N_TAG, m_N)
        if (name == NODE_TAG) {
            CNode node;
            if (traverser.traverseSubLevel(boost::bind(&CNode::acceptRestoreTraverser, &node, _1)) == false) {
                LOG_ERROR("Failed to restore NODE_TAG, got " << traverser.value());
            }
            if (nodeCount++ == 0) {
                m_Root = &m_NodeAllocator.create(node);
            } else {
                m_Root->insert(m_NodeAllocator, node);
            }
            continue;
        }
    } while (traverser.next());

    return true;
}

void CQDigest::add(uint32_t value, uint64_t n) {
    LOG_TRACE("Adding = " << value);

    m_N += n;

    CNode* expanded = m_Root->expand(m_NodeAllocator, value);
    if (expanded) {
        m_Root = expanded;
    }

    // If we already have the leaf node then incrementing a
    // leaf node count can't cause us to violate any constraints
    // so there is no need to compress unless we incremented
    // floor(n/k), in which case we need to compress the whole
    // tree. Otherwise, we can get away with just compressing
    // the path from the leaf to the root.

    CNode& leaf = m_Root->insert(m_NodeAllocator, CNode(value, value, n, n));
    if (expanded || (m_N / m_K) != ((m_N - n) / m_K)) {
        // Compress the whole tree.
        this->compress();
    } else if (leaf.count() == n) {
        // Compress the path to the new leaf.
        TNodePtrVec compress(1u, &leaf);
        this->compress(compress);
    }

    //this->checkInvariants();
}

void CQDigest::merge(const CQDigest& digest) {
    TNodePtrVec nodes;
    digest.m_Root->postOrder(nodes);

    CNode* expanded = m_Root->expand(m_NodeAllocator, digest.m_Root->max());
    if (expanded) {
        m_Root = expanded;
    }

    for (const auto& node : nodes) {
        m_N += node->count();
        m_Root->insert(m_NodeAllocator, *node);
    }

    // Compress the whole tree.
    this->compress();

    //this->checkInvariants();
}

void CQDigest::propagateForwardsByTime(double time) {
    if (time < 0.0) {
        LOG_ERROR("Can't propagate quantiles backwards in time");
        return;
    }

    double alpha = ::exp(-m_DecayRate * time);

    m_N = m_Root->age(alpha);

    // Compress the whole tree.
    this->compress();
}

bool CQDigest::scale(double factor) {
    typedef boost::tuple<uint32_t, uint32_t, uint64_t> TUInt32UInt32UInt64Tr;
    typedef std::vector<TUInt32UInt32UInt64Tr> TUInt32UInt32UInt64TrVec;

    if (factor <= 0.0) {
        LOG_ERROR("Scaling factor must be positive");
        return false;
    }

    if (factor == 1.0) {
        // Nothing to do.
        return true;
    }

    if (m_N == 0) {
        // Nothing to do.
        return true;
    }

    // Get a sketch of the current q-digest.
    TNodePtrVec nodes;
    m_Root->postOrder(nodes);
    std::sort(nodes.begin(), nodes.end(), SLevelLess());
    TUInt32UInt32UInt64TrVec sketch;
    sketch.reserve(nodes.size());
    for (const auto& node : nodes) {
        sketch.emplace_back(node->min(), node->max(), node->count());
    }

    // Start again from scratch.
    this->clear();

    // Reinsert the scaled summary values.
    boost::random::mt11213b generator;
    for (std::size_t i = 0u; i < sketch.size(); ++i) {
        const TUInt32UInt32UInt64Tr& node = sketch[i];

        uint32_t min = node.get<0>();
        uint32_t max = node.get<1>();
        uint32_t span = max - min + 1;
        uint64_t count = node.get<2>() / span;
        uint64_t remainder = node.get<2>() - count * span;
        LOG_TRACE("min = " << min << ", max = " << max << ", count = " << count << ", remainder = " << remainder);

        if (count > 0) {
            for (uint32_t j = 0u; j < span; ++j) {
                this->add(static_cast<uint32_t>(factor * static_cast<double>(min + j) + 0.5), count);
            }
        }
        if (remainder > 0) {
            boost::random::uniform_int_distribution<uint32_t> uniform(0u, span - 1);
            for (uint64_t j = 0u; j < remainder; ++j) {
                this->add(static_cast<uint32_t>(factor * static_cast<double>(min + uniform(generator)) + 0.5));
            }
        }
    }

    return true;
}

void CQDigest::clear(void) {
    // Release all current nodes.
    TNodePtrVec nodes;
    m_Root->postOrder(nodes);
    for (const auto& node : nodes) {
        m_N -= node->count();
    }

    // Reset root to its initial state and sanity check total count.
    m_Root = &m_NodeAllocator.create(CNode(0, 1, 0, 0));
    if (m_N != 0) {
        LOG_ERROR("Inconsistency - sum of node counts did not equal N");
        m_N = 0;
    }
}

bool CQDigest::quantile(double q, uint32_t& result) const {
    result = 0u;

    if (m_N == 0) {
        LOG_ERROR("Can't compute quantiles on empty set");
        return false;
    }

    // Compute the count fraction we need to the left of the value.
    uint64_t n = static_cast<uint64_t>(q * static_cast<double>(m_N) + 0.5);

    result = m_Root->quantile(0, n);

    return true;
}

bool CQDigest::quantileSublevelSetSupremum(double f, uint32_t& result) const {
    result = 0;
    if (m_N == 0) {
        LOG_ERROR("Can't compute level set for empty set");
        return false;
    }
    if (f <= 0.0) {
        m_Root->sublevelSetSupremum(-1, result);
        return true;
    }
    if (f > 1.0) {
        m_Root->superlevelSetInfimum(m_Root->max() + 1, result);
        return true;
    }

    uint64_t n = static_cast<uint64_t>(f * static_cast<double>(m_N) + 0.5);
    m_Root->quantileSublevelSetSupremum(n, 0, result);
    return true;
}

double CQDigest::cdfQuantile(double n, double p, double q) {
    if (q == 0.5) {
        return p;
    }

    // This accounts for the fact that we have a finite sample size
    // when computing the fraction f corresponding to a c.d.f value
    // of p.
    //
    // We use a Bayesian approach. The count of events with fraction
    // f is binomially distributed. Therefore, if we assume a non-
    // informative beta prior we can get a posterior distribution
    // for the true fraction simply by setting:
    //   alpha = alpha + p * n
    //   beta  = beta + n * (1 - p).

    static const double ONE_THIRD = 1.0 / 3.0;

    try {
        double a = n * p + ONE_THIRD;
        double b = n * (1.0 - p) + ONE_THIRD;
        boost::math::beta_distribution<> beta(a, b);
        return boost::math::quantile(beta, q);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to calculate c.d.f. quantile: " << e.what() << ", n = " << n << ", p = " << p << ", q = " << q);
    }
    return p;
}

bool CQDigest::cdf(uint32_t x, double confidence, double& lowerBound, double& upperBound) const {
    lowerBound = 0.0;
    upperBound = 0.0;

    if (m_N == 0) {
        LOG_ERROR("Can't compute c.d.f. for empty set");
        return false;
    }

    uint64_t l = 0ull;
    m_Root->cdfLowerBound(x, l);
    lowerBound = static_cast<double>(l) / static_cast<double>(m_N);
    if (confidence > 0.0) {
        lowerBound = cdfQuantile(static_cast<double>(m_N), lowerBound, (100.0 - confidence) / 200.0);
    }

    uint64_t u = 0ull;
    m_Root->cdfUpperBound(x, u);
    upperBound = static_cast<double>(u) / static_cast<double>(m_N);
    if (confidence > 0.0) {
        upperBound = cdfQuantile(static_cast<double>(m_N), upperBound, (100.0 + confidence) / 200.0);
    }

    return true;
}

void CQDigest::pdf(uint32_t x, double confidence, double& lowerBound, double& upperBound) const {
    lowerBound = 0.0;
    upperBound = 0.0;

    if (m_N == 0) {
        return;
    }

    uint32_t infimum = 0u;
    m_Root->superlevelSetInfimum(x, infimum);

    uint32_t supremum = std::numeric_limits<uint32_t>::max();
    m_Root->sublevelSetSupremum(static_cast<int64_t>(x), supremum);

    double infimumLowerBound;
    double infimumUpperBound;
    this->cdf(infimum, confidence, infimumLowerBound, infimumUpperBound);

    double supremumLowerBound;
    double supremumUpperBound;
    this->cdf(supremum, confidence, supremumLowerBound, supremumUpperBound);

    lowerBound = std::max(supremumLowerBound - infimumUpperBound, 0.0) / std::max(static_cast<double>(supremum - infimum), 1.0);
    upperBound = std::max(supremumUpperBound - infimumLowerBound, 0.0) / std::max(static_cast<double>(supremum - infimum), 1.0);

    LOG_TRACE("x = " << x << ", supremum = " << supremum << ", infimum = " << infimum << ", cdf(supremum) = [" << supremumLowerBound << ","
                     << supremumUpperBound << "]"
                     << ", cdf(infimum) = [" << infimumLowerBound << "," << infimumUpperBound << "]"
                     << ", pdf = [" << lowerBound << "," << upperBound << "]");
}

void CQDigest::sublevelSetSupremum(uint32_t x, uint32_t& result) const {
    m_Root->sublevelSetSupremum(static_cast<int64_t>(x), result);
}

void CQDigest::superlevelSetInfimum(uint32_t x, uint32_t& result) const {
    m_Root->superlevelSetInfimum(x, result);
}

void CQDigest::summary(TUInt32UInt64PrVec& result) const {
    result.clear();

    if (m_N == 0) {
        return;
    }

    TNodePtrVec nodes;
    m_Root->postOrder(nodes);

    result.reserve(nodes.size());

    uint32_t last = nodes[0]->max();
    uint64_t count = nodes[0]->count();
    for (std::size_t i = 1u; i < nodes.size(); ++i) {
        if (nodes[i]->max() != last) {
            result.emplace_back(last, count);
            last = nodes[i]->max();
        }

        count += nodes[i]->count();
    }

    // Check if any count is aligned with the root max.
    if (result.empty() || result.back().second < count) {
        result.emplace_back(m_Root->max(), count);
    }

    if (result.back().second != m_N) {
        LOG_ERROR("Got " << result.back().second << " expected " << m_N);
    }
}

uint64_t CQDigest::n(void) const {
    return m_N;
}

uint64_t CQDigest::k(void) const {
    return m_K;
}

uint64_t CQDigest::checksum(uint64_t seed) const {
    seed = CChecksum::calculate(seed, m_K);
    seed = CChecksum::calculate(seed, m_N);
    seed = CChecksum::calculate(seed, m_DecayRate);
    TUInt32UInt64PrVec summary;
    this->summary(summary);
    return CChecksum::calculate(seed, summary);
}

bool CQDigest::checkInvariants(void) const {
    // These are:
    //   1) |Q| <= 3 * k.
    //   2) Subtree count at the root = n
    //   2) The node invariants are satisfied.

    if (m_Root->size() > 3 * m_K) {
        LOG_ERROR("|Q| = " << m_Root->size() << " 3k = " << 3 * m_K);
        return false;
    }

    if (m_Root->subtreeCount() != m_N) {
        LOG_ERROR("Bad count: " << m_Root->subtreeCount() << ", n = " << m_N);
        return false;
    }

    return m_Root->checkInvariants(m_N / m_K);
}

std::string CQDigest::print(void) const {
    std::ostringstream result;

    TNodePtrVec nodes;
    m_Root->postOrder(nodes);

    result << m_N << " | " << m_K << " | {";
    for (const auto& node : nodes) {
        result << " \"" << node->print() << ',' << node->count() << ',' << node->subtreeCount() << '"';
    }
    result << " }";

    return result.str();
}

void CQDigest::compress(void) {
    for (std::size_t i = 0u; i < 3 * m_K + 2; ++i) {
        TNodePtrVec compress;
        m_Root->postOrder(compress);
        if (!this->compress(compress)) {
            return;
        }
    }
    LOG_ERROR("Failed to compress tree");
}

bool CQDigest::compress(TNodePtrVec& compress) {
    bool compressed = false;

    std::make_heap(compress.begin(), compress.end(), SLevelLess());

    while (!compress.empty()) {
        CNode& node = *compress.front();

        std::pop_heap(compress.begin(), compress.end(), SLevelLess());
        compress.pop_back();

        if (CNode* parent = node.compress(m_NodeAllocator, m_N / m_K)) {
            compressed = true;

            compress.push_back(parent);
            std::push_heap(compress.begin(), compress.end(), SLevelLess());
        }
    }

    return compressed;
}

bool CQDigest::SLevelLess::operator()(const CNode* lhs, const CNode* rhs) const {
    return lhs->span() > rhs->span() || (lhs->span() == rhs->span() && lhs->max() > rhs->max());
}

bool CQDigest::SPostLess::operator()(const CNode* lhs, const CNode* rhs) const {
    return lhs->max() < rhs->max() || (lhs->max() == rhs->max() && lhs->span() < rhs->span());
}

const std::string CQDigest::CNode::MIN_TAG("a");
const std::string CQDigest::CNode::MAX_TAG("b");
const std::string CQDigest::CNode::COUNT_TAG("c");

CQDigest::CNode::CNode(void)
    : m_Ancestor(0), m_Descendants(), m_Min(0xDEADBEEF), m_Max(0xDEADBEEF), m_Count(0xDEADBEEF), m_SubtreeCount(0xDEADBEEF) {
}

CQDigest::CNode::CNode(uint32_t min, uint32_t max, uint64_t count, uint64_t subtreeCount)
    : m_Ancestor(0), m_Descendants(), m_Min(min), m_Max(max), m_Count(count), m_SubtreeCount(subtreeCount) {
}

std::size_t CQDigest::CNode::size(void) const {
    std::size_t size = 1u;

    for (const auto& descendant : m_Descendants) {
        size += descendant->size();
    }

    return size;
}

uint32_t CQDigest::CNode::quantile(uint64_t leftCount, uint64_t n) const {
    // We need to find the smallest node in post-order where
    // the left count is greater than n. At each level we visit
    // the smallest, in post order, node in the q-digest for
    // which the left count is greater than n. Terminating when
    // this node doesn't have any descendants.

    for (const auto& descendant : m_Descendants) {
        uint64_t count = descendant->subtreeCount();
        if (leftCount + count >= n) {
            return descendant->quantile(leftCount, n);
        }
        leftCount += count;
    }

    return m_Max;
}

bool CQDigest::CNode::quantileSublevelSetSupremum(uint64_t n, uint64_t leftCount, uint32_t& result) const {
    // We are looking for the right end of the rightmost node
    // whose count together with those nodes to the left is
    // is less than n.

    if (leftCount + m_SubtreeCount < n) {
        result = std::max(result, m_Max);
        return true;
    }

    leftCount += m_SubtreeCount;
    for (auto i = m_Descendants.rbegin(); i != m_Descendants.rend(); ++i) {
        leftCount -= (*i)->subtreeCount();
        if (leftCount + (*i)->count() < n && (*i)->quantileSublevelSetSupremum(n, leftCount, result)) {
            break;
        }
    }

    return false;
}

void CQDigest::CNode::cdfLowerBound(uint32_t x, uint64_t& result) const {
    // The lower bound is the sum of the counts at the nodes
    // for which the maximum value is less than or equal to x.

    if (m_Max <= x) {
        result += m_SubtreeCount;
    } else {
        for (const auto& descendant : m_Descendants) {
            descendant->cdfLowerBound(x, result);
        }
    }
}

void CQDigest::CNode::cdfUpperBound(uint32_t x, uint64_t& result) const {
    // The upper bound is the sum of the counts at the nodes
    // for which the minimum value is less than or equal to x.

    if (m_Max <= x) {
        result += m_SubtreeCount;
    } else if (m_Min <= x) {
        result += m_Count;
        for (const auto& descendant : m_Descendants) {
            descendant->cdfUpperBound(x, result);
        }
    }
}

void CQDigest::CNode::sublevelSetSupremum(const int64_t x, uint32_t& result) const {
    for (auto i = m_Descendants.rbegin(); i != m_Descendants.rend(); ++i) {
        if (static_cast<int64_t>((*i)->max()) > x) {
            result = std::min(result, (*i)->max());
        } else {
            (*i)->sublevelSetSupremum(x, result);
            break;
        }
    }
    if (static_cast<int64_t>(m_Max) > x && m_Count > 0) {
        result = std::min(result, m_Max);
    }
}

void CQDigest::CNode::superlevelSetInfimum(uint32_t x, uint32_t& result) const {
    for (const auto& descendant : m_Descendants) {
        if (descendant->max() < x) {
            result = std::max(result, descendant->max());
        } else {
            descendant->superlevelSetInfimum(x, result);
            break;
        }
    }
    if (m_Max < x && m_Count > 0) {
        result = std::max(result, m_Max);
    }
}

void CQDigest::CNode::postOrder(TNodePtrVec& nodes) const {
    for (const auto& descendant : m_Descendants) {
        descendant->postOrder(nodes);
    }
    nodes.push_back(const_cast<CNode*>(this));
}

CQDigest::CNode* CQDigest::CNode::expand(CNodeAllocator& allocator, const uint32_t& value) {
    if (m_Max >= value) {
        // No expansion necessary.
        return 0;
    }

    CNode* result = m_Count == 0 ? this : &allocator.create(CNode(m_Min, m_Max, 0, 0));
    uint32_t levelSpan = result->span();
    do {
        result->m_Max += levelSpan;
        levelSpan <<= 1;
    } while (result->m_Max < value);

    if (result != this) {
        m_Ancestor = result;
        result->m_Descendants.push_back(this);
        result->m_SubtreeCount += m_SubtreeCount;
    }

    return result;
}

CQDigest::CNode& CQDigest::CNode::insert(CNodeAllocator& allocator, const CNode& node) {
    m_SubtreeCount += node.subtreeCount();

    if (*this == node) {
        m_Count += node.count();
        return *this;
    }

    auto next = std::lower_bound(m_Descendants.begin(), m_Descendants.end(), &node, SPostLess());

    // If it exists the ancestor will be after the node
    // in post order.
    for (auto i = next; i != m_Descendants.end(); ++i) {
        if ((*i)->isAncestor(node) || **i == node) {
            return (*i)->insert(allocator, node);
        }
    }

    // This is the lowest ancestor in the q-digest. Insert
    // the node below it in post order and move descendants
    // if necessary.
    CNode& newNode = allocator.create(node);
    newNode.m_Ancestor = this;
    m_Descendants.insert(next, &newNode);
    if (!newNode.isLeaf()) {
        newNode.takeDescendants(*this);
    }

    return newNode;
}

CQDigest::CNode* CQDigest::CNode::compress(CNodeAllocator& allocator, uint64_t compressionFactor) {
    if (!m_Ancestor) {
        // The node is no longer in the q-digest.
        return 0;
    }

    // Warning this function zeros m_Ancestor copy up front.
    CNode* ancestor = m_Ancestor;

    // Get the sibling of this node if it exists.
    CNode* sibling = ancestor->sibling(*this);

    uint64_t count = (ancestor->isParent(*this) ? ancestor->count() : 0ull) + this->count() + (sibling ? sibling->count() : 0ull);

    // Check if we should compress this node.
    if (count >= compressionFactor) {
        return 0;
    }

    if (ancestor->isParent(*this)) {
        ancestor->m_Count = count;
        this->detach(allocator);
        if (sibling) {
            sibling->detach(allocator);
        }
        return ancestor;
    }

    // We'll recycle this node for the parent.

    m_Count = count;
    this->isLeftChild() ? m_Max += this->span() : m_Min -= this->span();
    this->takeDescendants(*ancestor);
    if (sibling) {
        sibling->detach(allocator);
    }

    return this;
}

uint64_t CQDigest::CNode::age(double factor) {
    m_SubtreeCount = 0u;

    for (auto&& descendant : m_Descendants) {
        m_SubtreeCount += descendant->age(factor);
    }

    if (m_Count > 0) {
        m_Count = static_cast<uint64_t>(std::max(static_cast<double>(m_Count) * factor + 0.5, 1.0));
    }
    m_SubtreeCount += m_Count;

    return m_SubtreeCount;
}

uint32_t CQDigest::CNode::span(void) const {
    return m_Max - m_Min + 1u;
}

uint32_t CQDigest::CNode::min(void) const {
    return m_Min;
}

uint32_t CQDigest::CNode::max(void) const {
    return m_Max;
}

const uint64_t& CQDigest::CNode::count(void) const {
    return m_Count;
}

const uint64_t& CQDigest::CNode::subtreeCount(void) const {
    return m_SubtreeCount;
}

void CQDigest::CNode::persistRecursive(const std::string& nodeTag, core::CStatePersistInserter& inserter) const {
    inserter.insertLevel(NODE_TAG, boost::bind(&CNode::acceptPersistInserter, this, _1));

    // Note the tree is serialized flat in pre-order.
    for (const auto& descendant : m_Descendants) {
        descendant->persistRecursive(nodeTag, inserter);
    }
}

void CQDigest::CNode::acceptPersistInserter(core::CStatePersistInserter& inserter) const {
    inserter.insertValue(MIN_TAG, m_Min);
    inserter.insertValue(MAX_TAG, m_Max);
    inserter.insertValue(COUNT_TAG, m_Count);
}

bool CQDigest::CNode::acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) {
    do {
        const std::string& name = traverser.name();
        if (name == MIN_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_Min) == false) {
                LOG_ERROR("Invalid min in " << traverser.value());
                return false;
            }
        }
        if (name == MAX_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_Max) == false) {
                LOG_ERROR("Invalid max in " << traverser.value());
                return false;
            }
        }
        if (name == COUNT_TAG) {
            if (core::CStringUtils::stringToType(traverser.value(), m_Count) == false) {
                LOG_ERROR("Invalid count in " << traverser.value());
                return false;
            }
            m_SubtreeCount = m_Count;
        }
    } while (traverser.next());

    return true;
}

bool CQDigest::CNode::checkInvariants(uint64_t compressionFactor) const {
    // 1) span is a power of 2
    // 2) q-digest connectivity is consistent.
    // 3) subtree counts are consistent.
    // 4) if node != leaf count(node) <= floor(n/k)
    // 5) count(parent) + count(left) + count(right) > floor(n/k)

    // Subtracting 1 will flip all the bits to the right of the last 1 in the
    // current binary representation.  If the span is a power of 2 then it will
    // only have 1 bit set, so span minus 1 will have completely different bits
    // set to span.  Then an inclusive OR and an exclusive OR of span and span
    // minus 1 will be identical.  If span is not a power of 2 then subtracting
    // 1 will leave some set bits set, meaning the OR and XOR give different
    // results.
    uint32_t span(this->span());
    uint32_t spanMinusOne(span - 1);
    if ((span | spanMinusOne) != (span ^ spanMinusOne)) {
        LOG_ERROR("Bad span: " << this->print());
        return false;
    }

    SPostLess postLess;
    uint64_t subtreeCount = m_Count;

    for (std::size_t i = 0u; i < m_Descendants.size(); ++i) {
        if (m_Descendants[i]->m_Ancestor != this) {
            LOG_ERROR("Bad connectivity: " << this->print() << " -> " << m_Descendants[i]->print() << " <- "
                                           << m_Descendants[i]->m_Ancestor->print());
        }
        if (!this->isAncestor(*m_Descendants[i])) {
            LOG_ERROR("Bad connectivity: " << this->print() << " -> " << m_Descendants[i]->print());
            return false;
        }
        if (i + 1u < m_Descendants.size() && !postLess(m_Descendants[i], m_Descendants[i + 1u])) {
            LOG_ERROR("Bad order: " << m_Descendants[i]->print() << " >= " << m_Descendants[i + 1u]->print());
            return false;
        }
        if (!m_Descendants[i]->checkInvariants(compressionFactor)) {
            return false;
        }
        subtreeCount += m_Descendants[i]->subtreeCount();
    }

    if (subtreeCount != m_SubtreeCount) {
        LOG_ERROR("Bad subtree count: expected " << subtreeCount << " got " << m_SubtreeCount);
        return false;
    }

    if (!this->isLeaf() && !this->isRoot() && m_Count > compressionFactor) {
        LOG_ERROR("Bad count: " << m_Count << ", floor(n/k) = " << compressionFactor);
        return false;
    }

    if (!this->isRoot()) {
        const CNode* sibling = m_Ancestor->sibling(*this);
        uint64_t count = m_Count + (sibling ? sibling->count() : 0ull) + (m_Ancestor->isParent(*this) ? m_Ancestor->count() : 0ull);
        if (count < compressionFactor) {
            LOG_ERROR("Bad triple count: " << count << ", floor(n/k) = " << compressionFactor);
            return false;
        }
    }

    return true;
}

std::string CQDigest::CNode::print(void) const {
    std::ostringstream result;
    result << '[' << m_Min << ',' << m_Max << ']';
    return result.str();
}

bool CQDigest::CNode::operator==(const CNode& node) const {
    return m_Min == node.m_Min && m_Max == node.m_Max;
}

std::size_t CQDigest::CNode::numberDescendants(void) const {
    return m_Descendants.size();
}

CQDigest::TNodePtrVecCItr CQDigest::CNode::beginDescendants(void) const {
    return m_Descendants.begin();
}

CQDigest::TNodePtrVecCItr CQDigest::CNode::endDescendants(void) const {
    return m_Descendants.end();
}

CQDigest::CNode* CQDigest::CNode::sibling(const CNode& node) const {
    uint32_t min = node.min();
    node.isLeftChild() ? min += node.span() : min -= node.span();
    uint32_t max = node.max();
    node.isLeftChild() ? max += node.span() : max -= node.span();
    CNode sibling(min, max, 0u, 0u);

    auto next = std::lower_bound(m_Descendants.begin(), m_Descendants.end(), &sibling, SPostLess());

    if (next != m_Descendants.end() && (*next)->isSibling(node)) {
        return *next;
    }

    return 0;
}

bool CQDigest::CNode::isSibling(const CNode& node) const {
    // Check if the nodes are on the same level and share a parent.
    return this->span() == node.span() && (this->isLeftChild() ? m_Max + 1u == node.m_Min : m_Min == node.m_Max + 1u);
}

bool CQDigest::CNode::isParent(const CNode& node) const {
    // Check is ancestor and is in level above.
    return this->isAncestor(node) && this->span() == 2 * node.span();
}

bool CQDigest::CNode::isAncestor(const CNode& node) const {
    // Check for inclusion of node range.
    return (m_Min < node.m_Min && m_Max >= node.m_Max) || (m_Min <= node.m_Min && m_Max > node.m_Max);
}

bool CQDigest::CNode::isRoot(void) const {
    return m_Ancestor == 0;
}

bool CQDigest::CNode::isLeaf(void) const {
    return this->span() == 1;
}

bool CQDigest::CNode::isLeftChild(void) const {
    // The left child nodes are always an even multiple of the
    // level range from the start of the overall range and the
    // right child nodes an odd multiple. To reduce storage we
    // pass in the start of the overall range.

    return (m_Min / this->span()) % 2 == 0;
}

void CQDigest::CNode::detach(CNodeAllocator& allocator) {
    m_Ancestor->removeDescendant(*this);
    m_Ancestor->takeDescendants(*this);
    m_Ancestor = 0;
    allocator.release(*this);
}

void CQDigest::CNode::removeDescendant(CNode& node) {
    // Remove node from the descendants.
    m_Descendants.erase(std::remove(m_Descendants.begin(), m_Descendants.end(), &node), m_Descendants.end());
}

bool CQDigest::CNode::takeDescendants(CNode& node) {
    if (node.numberDescendants() == 0) {
        return false;
    }

    if (!this->isAncestor(node)) {
        // Find our descendants among the descendants of node.
        TNodePtrVec nodesToTake;
        TNodePtrVec nodesToLeave;
        for (auto i = node.beginDescendants(); i != node.endDescendants(); ++i) {
            if (this->isAncestor(**i)) {
                nodesToTake.push_back(*i);
                (*i)->m_Ancestor = this;
                m_SubtreeCount += (*i)->subtreeCount();
            } else {
                nodesToLeave.push_back(*i);
            }
        }

        // Merge the descendants.
        TNodePtrVec descendants;
        descendants.reserve(m_Descendants.size() + nodesToTake.size());
        std::merge(m_Descendants.begin(),
                   m_Descendants.end(),
                   nodesToTake.begin(),
                   nodesToTake.end(),
                   std::back_inserter(descendants),
                   SPostLess());

        // Update the node's descendants.
        nodesToLeave.swap(node.m_Descendants);

        // Write the result back to this node.
        descendants.swap(m_Descendants);

        return !nodesToTake.empty();
    }

    for (auto i = node.beginDescendants(); i != node.endDescendants(); ++i) {
        (*i)->m_Ancestor = this;
    }

    // Merge the descendants.
    TNodePtrVec descendants;
    descendants.reserve(m_Descendants.size() + node.numberDescendants());
    std::merge(m_Descendants.begin(),
               m_Descendants.end(),
               node.beginDescendants(),
               node.endDescendants(),
               std::back_inserter(descendants),
               SPostLess());

    // Clear out the node's descendants.
    TNodePtrVec empty;
    empty.swap(node.m_Descendants);

    // Write the result back to this node.
    descendants.swap(m_Descendants);

    return true;
}

CQDigest::CNodeAllocator::CNodeAllocator(std::size_t size) {
    m_Nodes.push_back(TNodeVec());
    m_Nodes.back().reserve(size);
    m_FreeNodes.push_back(TNodePtrVec());
}

CQDigest::CNode& CQDigest::CNodeAllocator::create(const CNode& node) {
    if (m_FreeNodes.front().empty()) {
        // Add a new collection if necessary. This should
        // only happen when merging two q-digests.
        std::size_t size = m_Nodes.back().size();
        if (size == m_Nodes.back().capacity()) {
            m_Nodes.push_back(TNodeVec());
            m_Nodes.back().reserve(size);
            m_FreeNodes.push_back(TNodePtrVec());
            LOG_TRACE("Added new block " << m_Nodes.size());
        }

        TNodeVec& nodes = m_Nodes.back();
        nodes.resize(nodes.size() + 1u);
        nodes.back() = node;
        return nodes.back();
    }

    CNode* freeNode = m_FreeNodes.front().back();
    *freeNode = node;
    m_FreeNodes.front().pop_back();
    return *freeNode;
}

void CQDigest::CNodeAllocator::release(CNode& node) {
    std::size_t block = this->findBlock(node);
    if (block >= m_FreeNodes.size()) {
        LOG_ABORT("Bad block address = " << block << ", max = " << m_FreeNodes.size() - 1u);
    }

    m_FreeNodes[block].push_back(&node);

    if (m_Nodes.size() > 1u) {
        auto nodeItr = m_Nodes.begin();
        std::advance(nodeItr, block);

        // Remove the block if none of its nodes are in use.
        if (m_FreeNodes[block].size() > nodeItr->size()) {
            LOG_TRACE("Removing block " << block);
            m_FreeNodes.erase(m_FreeNodes.begin() + block);
            m_Nodes.erase(nodeItr);
        }
    }
}

std::size_t CQDigest::CNodeAllocator::findBlock(const CNode& node) const {
    std::size_t result = 0u;

    if (m_Nodes.size() == 1u) {
        return result;
    }

    const auto le = std::less_equal<const CNode*>();

    for (auto i = m_Nodes.begin(); i != m_Nodes.end(); ++i, ++result) {
        auto first = i->begin();
        auto last = i->end();
        if (first == last) {
            continue;
        }

        --last;
        if (le(&(*first), &node) && le(&node, &(*last))) {
            break;
        }
    }

    return result;
}
}
}
