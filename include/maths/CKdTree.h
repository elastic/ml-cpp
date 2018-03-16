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

#ifndef INCLUDED_ml_maths_CKdTree_h
#define INCLUDED_ml_maths_CKdTree_h

#include <core/CLogger.h>

#include <maths/CAnnotatedVector.h>
#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CTypeConversions.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace ml {
namespace maths {

namespace kdtree_detail {

//! \brief Stubs out the node data parameter for k-d tree.
struct SEmptyNodeData {};

//! Euclidean norm implementation for our internal vector classes.
//!
//! Overload to adapt the Euclidean norm calculation for different
//! point implementations.
template<typename POINT>
typename SPromoted<typename SCoordinate<POINT>::Type>::Type euclidean(const POINT& point) {
    return point.euclidean();
}

} // kdtree_detail::

//! \brief A k-d tree.
//!
//! DESCRIPTION:\n
//! \see http://en.wikipedia.org/wiki/K-d_tree
//!
//! IMPLEMENTATION DECISIONS:\n
//! Our principle use case is to build the tree once up front and then
//! use it to make fast nearest neighbour queries. As such, the nodes
//! can be stored by value in pre-reserved vector, which is as efficient
//! as possible for this use case.
//!
//! The POINT type must have value semantics, support coordinate access,
//! via operator(), support subtraction, via operator-, and provide a
//! Euclidean norm, i.e. \f$\sqrt{\sum_i x_i^2}\f$, this is handled by
//! overloading the kdtree_detail::euclidean functions.
//!
//! Extra node data can be supplied as a template parameter. The nodes
//! inherit this publicly (so that the Empty Base Optimization is used
//! when it is not needed). This should be default constructible and
//! have value semantics. This can be useful for implementing certain
//! algorithms efficiently.
template<typename POINT, typename NODE_DATA = kdtree_detail::SEmptyNodeData>
class CKdTree {
public:
    typedef std::vector<POINT> TPointVec;
    typedef typename TPointVec::iterator TPointVecItr;
    typedef typename SCoordinate<POINT>::Type TCoordinate;
    typedef typename SPromoted<TCoordinate>::Type TCoordinatePrecise;
    typedef std::pair<TCoordinatePrecise, POINT> TCoordinatePrecisePointPr;
    typedef CBasicStatistics::COrderStatisticsHeap<TCoordinatePrecisePointPr> TNearestAccumulator;

    //! Less on a specific coordinate of point position vector.
    class CCoordinateLess {
    public:
        CCoordinateLess(std::size_t i) : m_I(i) {}
        bool operator()(const POINT& lhs, const POINT& rhs) const { return lhs(m_I) < rhs(m_I); }

    private:
        std::size_t m_I;
    };

    //! A node of the k-d tree.
    struct SNode : public NODE_DATA {
        SNode(SNode* parent, const POINT& point)
            : NODE_DATA(), s_Parent(parent), s_LeftChild(0), s_RightChild(0), s_Point(point) {}

        //! Check node invariants.
        bool checkInvariants(std::size_t dimension) const {
            if (s_Parent) {
                if (s_Parent->s_LeftChild != this && s_Parent->s_RightChild != this) {
                    LOG_ERROR("Not parent's child");
                    return false;
                }
            }

            std::size_t coordinate = this->depth() % dimension;
            CCoordinateLess less(coordinate);
            if (s_LeftChild && less(s_Point, s_LeftChild->s_Point)) {
                LOG_ERROR("parent = " << s_Point << ", left child = " << s_LeftChild->s_Point
                                      << ", coordinate = " << coordinate);
                return false;
            }
            if (s_RightChild && less(s_RightChild->s_Point, s_Point)) {
                LOG_ERROR("parent = " << s_Point << ", right child = " << s_RightChild->s_Point
                                      << ", coordinate = " << coordinate);
                return false;
            }
            return true;
        }

        //! Get the coordinate the points are split on.
        std::size_t depth(void) const {
            std::size_t depth = 0u;
            for (const SNode* ancestor = s_Parent; ancestor; ancestor = ancestor->s_Parent) {
                ++depth;
            }
            return depth;
        }

        //! The parent.
        SNode* s_Parent;
        //! The left child if one exists.
        SNode* s_LeftChild;
        //! The right child if one exists.
        SNode* s_RightChild;
        //! The point at this node.
        POINT s_Point;
    };

public:
    //! Reserve space for \p n points.
    void reserve(std::size_t n) { m_Nodes.reserve(n); }

    //! Build a k-d tree on the collection of points \p points.
    //!
    //! \note \p points are reordered by this operation.
    void build(TPointVec& points) {
        if (points.empty()) {
            return;
        }
        m_Dimension = points[0].dimension();
        m_Nodes.clear();
        m_Nodes.reserve(points.size());
        this->buildRecursively(0, // Parent pointer
                               0, // Split coordinate
                               points.begin(),
                               points.end());
    }

    //! Get the number of points in the tree.
    std::size_t size(void) const { return m_Nodes.size(); }

    //! Branch and bound search for nearest neighbour of \p point.
    const POINT* nearestNeighbour(const POINT& point) const {
        const POINT* nearest = 0;

        if (m_Nodes.empty()) {
            return nearest;
        }

        TCoordinatePrecise distanceToNearest = std::numeric_limits<TCoordinatePrecise>::max();
        return this->nearestNeighbour(point,
                                      m_Nodes[0],
                                      0, // Split coordinate,
                                      nearest,
                                      distanceToNearest);
    }

    //! Branch and bound search for nearest \p n neighbours of \p point.
    void nearestNeighbours(std::size_t n, const POINT& point, TPointVec& result) const {
        result.clear();

        if (n == 0 || m_Nodes.empty()) {
            return;
        }

        TNearestAccumulator nearest(n);
        this->nearestNeighbours(point,
                                m_Nodes[0],
                                0, // Split coordinate,
                                nearest);

        result.reserve(nearest.count());
        nearest.sort();
        for (std::size_t i = 0u; i < nearest.count(); ++i) {
            result.push_back(nearest[i].second);
        }
    }

    //! A pre-order depth first traversal of the k-d tree nodes.
    //!
    //! \param[in] f The function to apply to the nodes.
    //! \tparam F should have the signature bool (const SNode &).
    //! Traversal stops below point that \p f returns false.
    template<typename F>
    void preorderDepthFirst(F f) const {
        if (m_Nodes.empty()) {
            return;
        }
        this->preorderDepthFirst(m_Nodes[0], f);
    }

    //! A post-order depth first traversal of the k-d tree nodes.
    //!
    //! \param[in] f The function to apply to the nodes.
    //! \tparam F should have the signature void (const SNode &).
    template<typename F>
    void postorderDepthFirst(F f) const {
        if (m_Nodes.empty()) {
            return;
        }
        this->postorderDepthFirst(m_Nodes[0], f);
    }

    //! Check the tree invariants.
    bool checkInvariants(void) const {
        for (std::size_t i = 0u; i < m_Nodes.size(); ++i) {
            if (!m_Nodes[i].checkInvariants(m_Dimension)) {
                return false;
            }
        }
        return true;
    }

private:
    typedef std::vector<SNode> TNodeVec;

private:
    //! Recursively build the k-d tree.
    SNode* buildRecursively(SNode* parent, std::size_t coordinate, TPointVecItr begin, TPointVecItr end) {
        std::size_t n = static_cast<std::size_t>(end - begin) / 2;
        TPointVecItr median = begin + n;
        std::nth_element(begin, median, end, CCoordinateLess(coordinate));
        m_Nodes.push_back(SNode(parent, *median));
        SNode* node = &m_Nodes.back();
        if (median - begin > 0) {
            SNode* leftChild = this->buildRecursively(node, (coordinate + 1) % m_Dimension, begin, median);
            node->s_LeftChild = leftChild;
        }
        if (end - median > 1) {
            SNode* rightChild = this->buildRecursively(node, (coordinate + 1) % m_Dimension, median + 1, end);
            node->s_RightChild = rightChild;
        }
        return node;
    }

    //! Recursively find the nearest point to \p point.
    const POINT* nearestNeighbour(const POINT& point,
                                  const SNode& node,
                                  std::size_t coordinate,
                                  const POINT* nearest,
                                  TCoordinatePrecise& distanceToNearest) const {
        TCoordinatePrecise distance = kdtree_detail::euclidean(point - node.s_Point);

        if (distance < distanceToNearest) {
            nearest = &node.s_Point;
            distanceToNearest = distance;
        }

        if (node.s_LeftChild || node.s_RightChild) {
            TCoordinatePrecise distanceToHyperplane = point(coordinate) - node.s_Point(coordinate);

            SNode* primary = node.s_LeftChild;
            SNode* secondary = node.s_RightChild;
            if (!primary || (secondary && distanceToHyperplane > 0)) {
                std::swap(primary, secondary);
            }

            std::size_t nextCoordinate = (coordinate + 1) % m_Dimension;
            nearest = this->nearestNeighbour(point, *primary, nextCoordinate, nearest, distanceToNearest);
            if (secondary && ::fabs(distanceToHyperplane) < distanceToNearest) {
                nearest = this->nearestNeighbour(point, *secondary, nextCoordinate, nearest, distanceToNearest);
            }
        }

        return nearest;
    }

    //! Recursively find the nearest point to \p point.
    void nearestNeighbours(const POINT& point,
                           const SNode& node,
                           std::size_t coordinate,
                           TNearestAccumulator& nearest) const {
        TCoordinatePrecise distance = kdtree_detail::euclidean(point - node.s_Point);

        nearest.add(TCoordinatePrecisePointPr(distance, node.s_Point));

        if (node.s_LeftChild || node.s_RightChild) {
            TCoordinatePrecise distanceToHyperplane = point(coordinate) - node.s_Point(coordinate);

            SNode* primary = node.s_LeftChild;
            SNode* secondary = node.s_RightChild;
            if (!primary || (secondary && distanceToHyperplane > 0)) {
                std::swap(primary, secondary);
            }

            std::size_t nextCoordinate = (coordinate + 1) % m_Dimension;
            this->nearestNeighbours(point, *primary, nextCoordinate, nearest);
            if (secondary && ::fabs(distanceToHyperplane) < nearest.biggest().first) {
                this->nearestNeighbours(point, *secondary, nextCoordinate, nearest);
            }
        }
    }

    //! Visit the branch rooted at \p node with \p f in pre-order.
    template<typename F>
    static void preorderDepthFirst(const SNode& node, F f) {
        if (f(node)) {
            if (node.s_LeftChild) {
                preorderDepthFirst(*node.s_LeftChild, f);
            }
            if (node.s_RightChild) {
                preorderDepthFirst(*node.s_RightChild, f);
            }
        }
    }

    //! Visit the branch rooted at \p node with \p f in post-order.
    template<typename F>
    static void postorderDepthFirst(const SNode& node, F f) {
        if (node.s_LeftChild) {
            postorderDepthFirst(*node.s_LeftChild, f);
        }
        if (node.s_RightChild) {
            postorderDepthFirst(*node.s_RightChild, f);
        }
        f(node);
    }

private:
    //! The point dimension.
    std::size_t m_Dimension;
    //! The representation of the points.
    TNodeVec m_Nodes;
};
}
}

#endif // INCLUDED_ml_maths_CKdTree_h
