/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CKdTree_h
#define INCLUDED_ml_maths_CKdTree_h

#include <core/CLogger.h>

#include <maths/CAnnotatedVector.h>
#include <maths/CBasicStatistics.h>
#include <maths/CLinearAlgebra.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CLinearAlgebraTools.h>
#include <maths/COrderings.h>
#include <maths/CTypeTraits.h>

#include <boost/operators.hpp>
#include <boost/ref.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace ml {
namespace maths {

namespace kdtree_detail {

//! \brief Stubs out the node data parameter for k-d tree.
struct SEmptyNodeData {};

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
//! Euclidean norm, i.e. \f$\sqrt{\sum_i x_i^2}\f$. This is handled by
//! overloading the las::distance functions.
//!
//! Extra node data can be supplied as a template parameter. The nodes
//! inherit this publicly (so that the Empty Base Optimization is used
//! when it is not needed). This should be default constructible and
//! have value semantics. This can be useful for implementing certain
//! algorithms efficiently.
template<typename POINT, typename NODE_DATA = kdtree_detail::SEmptyNodeData>
class CKdTree {
public:
    using TDoubleVec = std::vector<double>;
    using TPointVec = std::vector<POINT>;
    using TCoordinate = typename SCoordinate<POINT>::Type;
    using TCoordinatePrecise = typename SPromoted<TCoordinate>::Type;
    using TPointCRef = boost::reference_wrapper<const POINT>;
    using TCoordinatePrecisePointCRefPr = std::pair<TCoordinatePrecise, TPointCRef>;
    using TCoordinatePrecisePointCRefPrVec = std::vector<TCoordinatePrecisePointCRefPr>;

    //! Less on a specific coordinate of point position vector.
    class CCoordinateLess {
    public:
        CCoordinateLess(std::size_t i) : m_I(i) {}
        bool operator()(const POINT& lhs, const POINT& rhs) const {
            return lhs(m_I) < rhs(m_I);
        }

    private:
        std::size_t m_I;
    };

    //! A node of the k-d tree.
    struct SNode : public NODE_DATA {
        //! Copy \p point into place.
        SNode(SNode* parent, const POINT& point)
            : s_Parent(parent), s_LeftChild(nullptr), s_RightChild(nullptr),
              s_Point(point) {}

        //! Move \p point into place.
        //!
        //! \note Since this is an internal class and we ensure that this
        //! constructor is called by explicitly calling std::move(point)
        //! in the version of append with move semantics we prefer to
        //! explicitly define a constructor taking an rvalue reference
        //! since not all our vector types can be moved and calling the
        //! other constructor saves us a redundant copy in these cases.
        SNode(SNode* parent, POINT&& point)
            : s_Parent(parent), s_LeftChild(nullptr), s_RightChild(nullptr),
              s_Point(std::forward<POINT>(point)) {}

        //! Check node invariants.
        bool checkInvariants(std::size_t dimension) const {
            if (s_Parent) {
                if (s_Parent->s_LeftChild != this && s_Parent->s_RightChild != this) {
                    LOG_ERROR(<< "Not parent's child");
                    return false;
                }
            }

            std::size_t coordinate{this->depth() % dimension};
            CCoordinateLess less(coordinate);
            if (s_LeftChild && less(s_Point, s_LeftChild->s_Point)) {
                LOG_ERROR(<< "parent = " << s_Point << ", left child = "
                          << s_LeftChild->s_Point << ", coordinate = " << coordinate);
                return false;
            }
            if (s_RightChild && less(s_RightChild->s_Point, s_Point)) {
                LOG_ERROR(<< "parent = " << s_Point << ", right child = "
                          << s_RightChild->s_Point << ", coordinate = " << coordinate);
                return false;
            }
            return true;
        }

        //! Get the coordinate the points are split on.
        std::size_t depth() const {
            std::size_t depth{0};
            for (const SNode* ancestor = s_Parent; ancestor; ancestor = ancestor->s_Parent) {
                ++depth;
            }
            return depth;
        }

        //! Estimate the amount of memory this node will use.
        static std::size_t estimateMemoryUsage(std::size_t dimension) {
            return sizeof(SNode) + las::estimateMemoryUsage<POINT>(dimension);
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
    using TNodeVec = std::vector<SNode>;
    using TNodeVecCItr = typename TNodeVec::const_iterator;

    //! \brief Iterates points in the tree.
    class TPointIterator
        : public boost::random_access_iterator_helper<TPointIterator, POINT, std::ptrdiff_t, const POINT*, const POINT&> {
    public:
        TPointIterator() = default;
        TPointIterator(TNodeVecCItr itr) : m_Itr(itr) {}
        const POINT& operator*() const { return m_Itr->s_Point; }
        const POINT* operator->() const { return &m_Itr->s_Point; }
        const POINT& operator[](std::ptrdiff_t n) { return m_Itr[n].s_Point; }
        bool operator==(const TPointIterator& rhs) const {
            return m_Itr == rhs.m_Itr;
        }
        bool operator<(const TPointIterator& rhs) const {
            return m_Itr < rhs.m_Itr;
        }
        TPointIterator& operator++() {
            ++m_Itr;
            return *this;
        }
        TPointIterator& operator--() {
            --m_Itr;
            return *this;
        }
        TPointIterator& operator+=(std::ptrdiff_t n) {
            m_Itr += n;
            return *this;
        }
        TPointIterator& operator-=(std::ptrdiff_t n) {
            m_Itr -= n;
            return *this;
        }
        std::ptrdiff_t operator-(const TPointIterator& rhs) const {
            return m_Itr - rhs.m_Itr;
        }

    private:
        TNodeVecCItr m_Itr;
    };

public:
    //! Reserve space for \p n points.
    void reserve(std::size_t n) { m_Nodes.reserve(n); }

    //! Build a k-d tree on the collection of points \p points.
    //!
    //! \note The vector \p points is reordered.
    void build(TPointVec& points) {
        this->build(points.begin(), points.end(), std::false_type{});
    }

    //! Build a k-d tree on the collection of points \p points.
    //!
    //! \note The \p points are moved into place.
    void build(TPointVec&& points) {
        // We forward to a local variable so the the space allocated for
        // the vector buffer is freed at the end of this function.
        TPointVec points_{std::forward<TPointVec>(points)};
        this->build(points_.begin(), points_.end(), std::true_type{});
    }

    //! Build from a pair of output random access iterators.
    //!
    //! \note The range [\p begin, \p end) is reordered.
    template<typename ITR, typename MOVE = std::false_type>
    void build(ITR begin, ITR end, MOVE move = MOVE{}) {
        if (begin == end) {
            return;
        }
        m_Dimension = las::dimension(*begin);
        m_Nodes.clear();
        m_Nodes.reserve(std::distance(begin, end));
        this->buildRecursively(nullptr, // Parent pointer
                               0,       // Split coordinate
                               begin, end, move);
    }

    //! Get the number of points in the tree.
    std::size_t size() const { return m_Nodes.size(); }

    //! Branch and bound search for nearest neighbour of \p point.
    const POINT* nearestNeighbour(const POINT& point) const {
        const POINT* nearest{nullptr};
        if (m_Nodes.size() > 0) {
            auto inf = std::numeric_limits<TCoordinatePrecise>::max();
            POINT distancesToHyperplanes{las::zero(point)};
            return this->nearestNeighbour(point, m_Nodes[0], distancesToHyperplanes,
                                          0 /*split coordinate*/, nearest, inf);
        }
        return nearest;
    }

    //! Branch and bound search for nearest \p n neighbours of \p point.
    void nearestNeighbours(std::size_t n, const POINT& point, TPointVec& result) const {

        result.clear();

        if (n > 0 && n < m_Nodes.size()) {
            auto inf = std::numeric_limits<TCoordinatePrecise>::max();

            // These neighbour points will be completely replaced by the call
            // to nearestNeighbours, but we need the collection to be initialized
            // with infinite distances so we get the correct value for the furthest
            // nearest neighbour at the start of the branch and bound search.
            COrderings::SLess less;
            POINT distancesToHyperplanes{las::zero(point)};
            TCoordinatePrecisePointCRefPrVec neighbours(
                n, {inf, boost::cref(m_Nodes[0].s_Point)});
            this->nearestNeighbours(point, less, m_Nodes[0], distancesToHyperplanes,
                                    0 /*split coordinate*/, neighbours);

            result.reserve(n);
            std::sort_heap(neighbours.begin(), neighbours.end(), less);
            for (const auto& neighbour : neighbours) {
                result.push_back(neighbour.second);
            }
        } else if (n > m_Nodes.size()) {
            TDoubleVec distances;
            distances.reserve(m_Nodes.size());
            result.reserve(m_Nodes.size());
            for (const auto& node : m_Nodes) {
                distances.push_back(las::distance(point, node.s_Point));
                result.push_back(node.s_Point);
            }
            COrderings::simultaneousSort(distances, result);
        }
    }

    //! Get an iterator over the points in the tree.
    TPointIterator begin() const { return TPointIterator(m_Nodes.begin()); }

    //! Get an iterator to the end of the points in the tree.
    TPointIterator end() const { return TPointIterator(m_Nodes.end()); }

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
    bool checkInvariants() const {
        for (const auto& node : m_Nodes) {
            if (node.checkInvariants(m_Dimension) == false) {
                return false;
            }
        }
        return true;
    }

    //! Estimate the amount of memory the k-d tree will use.
    //!
    //! \param[in] numberPoints The number of points it will hold.
    //! \param[in] dimension The dimension of points it will hold.
    static std::size_t estimateMemoryUsage(std::size_t numberPoints, std::size_t dimension) {
        return numberPoints * SNode::estimateMemoryUsage(dimension);
    }

private:
    //! Append a node moving \p point into place.
    void append(std::true_type, SNode* parent, POINT& point) {
        m_Nodes.emplace_back(parent, std::move(point));
    }

    //! Append a node copying \p point into place.
    void append(std::false_type, SNode* parent, const POINT& point) {
        m_Nodes.emplace_back(parent, point);
    }

    //! Recursively build the k-d tree.
    template<typename ITR, typename MOVE>
    SNode* buildRecursively(SNode* parent, std::size_t coordinate, ITR begin, ITR end, MOVE move) {
        std::size_t n{static_cast<std::size_t>(end - begin) / 2};
        ITR median{begin + n};
        std::nth_element(begin, median, end, CCoordinateLess(coordinate));
        this->append(move, parent, *median);
        SNode* node{&m_Nodes.back()};
        if (median - begin > 0) {
            std::size_t next{this->nextCoordinate(coordinate)};
            SNode* leftChild{this->buildRecursively(node, next, begin, median, move)};
            node->s_LeftChild = leftChild;
        }
        if (end - median > 1) {
            std::size_t next{this->nextCoordinate(coordinate)};
            SNode* rightChild{this->buildRecursively(node, next, median + 1, end, move)};
            node->s_RightChild = rightChild;
        }
        return node;
    }

    //! Recursively find the nearest point to \p point.
    const POINT* nearestNeighbour(const POINT& point,
                                  const SNode& node,
                                  POINT& distancesToHyperplanes,
                                  std::size_t coordinate,
                                  const POINT* nearest,
                                  TCoordinatePrecise& distanceToNearest) const {

        TCoordinatePrecise distance{las::distance(point, node.s_Point)};

        if (distance < distanceToNearest ||
            (distance == distanceToNearest && node.s_Point < point)) {
            distanceToNearest = distance;
            nearest = &node.s_Point;
        }

        const SNode* primary{node.s_LeftChild};
        const SNode* secondary{node.s_RightChild};

        if (primary != nullptr && secondary != nullptr) {
            TCoordinate distanceToHyperplane{point(coordinate) - node.s_Point(coordinate)};

            if (distanceToHyperplane > 0) {
                std::swap(primary, secondary);
            } else {
                distanceToHyperplane = std::fabs(distanceToHyperplane);
            }

            std::size_t nextCoordinate{this->nextCoordinate(coordinate)};
            nearest = this->nearestNeighbour(point, *primary, distancesToHyperplanes,
                                             nextCoordinate, nearest, distanceToNearest);
            std::swap(distancesToHyperplanes(coordinate), distanceToHyperplane);
            if (las::norm(distancesToHyperplanes) < distanceToNearest) {
                nearest = this->nearestNeighbour(point, *secondary,
                                                 distancesToHyperplanes, nextCoordinate,
                                                 nearest, distanceToNearest);
            }
            std::swap(distancesToHyperplanes(coordinate), distanceToHyperplane);
        } else if (primary != nullptr) {
            std::size_t nextCoordinate{this->nextCoordinate(coordinate)};
            nearest = this->nearestNeighbour(point, *primary, distancesToHyperplanes,
                                             nextCoordinate, nearest, distanceToNearest);
        } else if (secondary != nullptr) {
            std::size_t nextCoordinate{this->nextCoordinate(coordinate)};
            nearest = this->nearestNeighbour(point, *secondary, distancesToHyperplanes,
                                             nextCoordinate, nearest, distanceToNearest);
        }

        return nearest;
    }

    //! Recursively find the nearest point to \p point.
    void nearestNeighbours(const POINT& point,
                           const COrderings::SLess& less,
                           const SNode& node,
                           POINT& distancesToHyperplanes,
                           std::size_t coordinate,
                           TCoordinatePrecisePointCRefPrVec& nearest) const {

        TCoordinatePrecise distance{las::distance(point, node.s_Point)};

        if (distance < nearest.front().first ||
            (distance == nearest.front().first && node.s_Point < point)) {
            std::pop_heap(nearest.begin(), nearest.end(), less);
            nearest.back().first = distance;
            nearest.back().second = boost::cref(node.s_Point);
            std::push_heap(nearest.begin(), nearest.end(), less);
        }

        const SNode* primary{node.s_LeftChild};
        const SNode* secondary{node.s_RightChild};

        if (primary != nullptr && secondary != nullptr) {
            TCoordinate distanceToHyperplane{point(coordinate) - node.s_Point(coordinate)};

            if (distanceToHyperplane > 0) {
                std::swap(primary, secondary);
            } else {
                distanceToHyperplane = std::fabs(distanceToHyperplane);
            }

            std::size_t nextCoordinate{this->nextCoordinate(coordinate)};
            this->nearestNeighbours(point, less, *primary, distancesToHyperplanes,
                                    nextCoordinate, nearest);
            std::swap(distancesToHyperplanes(coordinate), distanceToHyperplane);
            if (las::norm(distancesToHyperplanes) < nearest.front().first) {
                this->nearestNeighbours(point, less, *secondary, distancesToHyperplanes,
                                        nextCoordinate, nearest);
            }
            std::swap(distancesToHyperplanes(coordinate), distanceToHyperplane);
        } else if (primary != nullptr) {
            std::size_t nextCoordinate{this->nextCoordinate(coordinate)};
            this->nearestNeighbours(point, less, *primary, distancesToHyperplanes,
                                    nextCoordinate, nearest);
        } else if (secondary != nullptr) {
            std::size_t nextCoordinate{this->nextCoordinate(coordinate)};
            this->nearestNeighbours(point, less, *secondary, distancesToHyperplanes,
                                    nextCoordinate, nearest);
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

    //! Get the next coordinate.
    inline std::size_t nextCoordinate(std::size_t coordinate) const {
        ++coordinate;
        // This branch works out significantly faster than modulo.
        return coordinate == m_Dimension ? 0 : coordinate;
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
