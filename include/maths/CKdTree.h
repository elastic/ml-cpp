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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ml
{
namespace maths
{

namespace kdtree_detail
{

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
template<typename POINT,
         typename NODE_DATA = kdtree_detail::SEmptyNodeData>
class CKdTree
{
    public:
        using TDoubleVec = std::vector<double>;
        using TPointVec = std::vector<POINT>;
        using TCoordinate = typename SCoordinate<POINT>::Type;
        using TCoordinatePrecise = typename SPromoted<TCoordinate>::Type;
        using TCoordinatePrecisePointPr = std::pair<TCoordinatePrecise, POINT>;
        using TNearestAccumulator = CBasicStatistics::COrderStatisticsHeap<TCoordinatePrecisePointPr>;

        //! Less on a specific coordinate of point position vector.
        class CCoordinateLess
        {
            public:
                CCoordinateLess(std::size_t i) : m_I(i) {}
                bool operator()(const POINT &lhs, const POINT &rhs) const
                {
                    return lhs(m_I) < rhs(m_I);
                }

            private:
                std::size_t m_I;
        };

        //! A node of the k-d tree.
        struct SNode : public NODE_DATA
        {
            //! Copy \p point into place.
            SNode(SNode *parent, const POINT &point) :
                s_Parent(parent),
                s_LeftChild(0),
                s_RightChild(0),
                s_Point(point)
            {}
            //! Move \p point into place.
            //!
            //! \note Since this is an internal class and we ensure that this
            //! constructor is called by explicitly calling std::move(point)
            //! in the version of append with move semantics we prefer to
            //! explicitly define a constructor taking an rvalue reference
            //! since not all our vector types can be moved and calling the
            //! other constructor saves us a redundant copy in these cases.
            SNode(SNode *parent, POINT &&point) :
                s_Parent(parent),
                s_LeftChild(0),
                s_RightChild(0),
                s_Point(std::move(point))
            {}

            //! Check node invariants.
            bool checkInvariants(std::size_t dimension) const
            {
                if (s_Parent)
                {
                    if (s_Parent->s_LeftChild != this && s_Parent->s_RightChild != this)
                    {
                        LOG_ERROR("Not parent's child");
                        return false;
                    }
                }

                std::size_t coordinate{this->depth() % dimension};
                CCoordinateLess less(coordinate);
                if (s_LeftChild && less(s_Point, s_LeftChild->s_Point))
                {
                    LOG_ERROR("parent = " << s_Point
                              << ", left child = " << s_LeftChild->s_Point
                              << ", coordinate = " << coordinate);
                    return false;
                }
                if (s_RightChild && less(s_RightChild->s_Point, s_Point))
                {
                    LOG_ERROR("parent = " << s_Point
                              << ", right child = " << s_RightChild->s_Point
                              << ", coordinate = " << coordinate);
                    return false;
                }
                return true;
            }

            //! Get the coordinate the points are split on.
            std::size_t depth(void) const
            {
                std::size_t depth{0u};
                for (const SNode *ancestor = s_Parent; ancestor; ancestor = ancestor->s_Parent)
                {
                    ++depth;
                }
                return depth;
            }

            //! The parent.
            SNode *s_Parent;
            //! The left child if one exists.
            SNode *s_LeftChild;
            //! The right child if one exists.
            SNode *s_RightChild;
            //! The point at this node.
            POINT s_Point;
        };
        using TNodeVec = std::vector<SNode>;
        using TNodeVecCItr = typename TNodeVec::const_iterator;

        //! \brief Iterates points in the tree.
        class TPointIterator : public std::iterator<std::forward_iterator_tag, const POINT>
        {
            public:
                TPointIterator(void) = default;
                TPointIterator(TNodeVecCItr itr) : m_Itr(itr) {}
                const POINT &operator*(void) const { return m_Itr->s_Point; }
                const POINT *operator->(void) const { return &m_Itr->s_Point; }
                bool operator==(const TPointIterator &rhs) const { return m_Itr == rhs.m_Itr; }
                bool operator!=(const TPointIterator &rhs) const { return m_Itr != rhs.m_Itr; }
                TPointIterator &operator++(void) { ++m_Itr; return *this; }
                TPointIterator operator++(int) { return m_Itr++; }

            private:
                TNodeVecCItr m_Itr;
        };

    private:
        //! Boolean true type.
        struct True {};
        //! Boolean false type.
        struct False {};

    public:
        //! Reserve space for \p n points.
        void reserve(std::size_t n)
        {
            m_Nodes.reserve(n);
        }

        //! Build a k-d tree on the collection of points \p points.
        //!
        //! \note The vector \p points is reordered.
        void build(TPointVec &points)
        {
            this->build(points.begin(), points.end());
        }

        //! Build a k-d tree on the collection of points \p points.
        //!
        //! \note The \p points are moved into place.
        void build(TPointVec &&points)
        {
            this->build(points.begin(), points.end(), True());
        }

        //! Build from a pair of output random access iterators.
        //!
        //! \note The range [\p begin, \p end) is reordered.
        template<typename ITR, typename MOVE = False>
        void build(ITR begin, ITR end, MOVE move = MOVE())
        {
            if (begin == end)
            {
                return;
            }
            m_Dimension = las::dimension(*begin);
            m_Nodes.clear();
            m_Nodes.reserve(std::distance(begin, end));
            this->buildRecursively(0, // Parent pointer
                                   0, // Split coordinate
                                   begin, end, move);
        }

        //! Get the number of points in the tree.
        std::size_t size(void) const
        {
            return m_Nodes.size();
        }

        //! Branch and bound search for nearest neighbour of \p point.
        const POINT *nearestNeighbour(const POINT &point) const
        {
            const POINT *nearest{0};

            if (!m_Nodes.empty())
            {
                TCoordinatePrecise distanceToNearest{
                        std::numeric_limits<TCoordinatePrecise>::max()};
                return this->nearestNeighbour(point,
                                              m_Nodes[0],
                                              0, // Split coordinate,
                                              nearest,
                                              distanceToNearest);
            }
            return nearest;
        }

        //! Branch and bound search for nearest \p n neighbours of \p point.
        void nearestNeighbours(std::size_t n,
                               const POINT &point,
                               TPointVec &result) const
        {
            result.clear();

            if (n > 0 && n < m_Nodes.size())
            {
                TNearestAccumulator neighbours(n);
                const TCoordinatePrecise inf{
                    boost::numeric::bounds<TCoordinatePrecise>::highest()};
                for (std::size_t i = 0u; i < n; ++i)
                {
                    neighbours.add({inf, las::zero(point)});
                }
                this->nearestNeighbours(point,
                                        m_Nodes[0],
                                        0, // Split coordinate,
                                        neighbours);

                result.reserve(neighbours.count());
                neighbours.sort();
                for (const auto &neighbour : neighbours)
                {
                    result.push_back(std::move(neighbour.second));
                }
            }
            else if (n > m_Nodes.size())
            {
                TDoubleVec distances;
                distances.reserve(m_Nodes.size());
                result.reserve(m_Nodes.size());
                for (const auto &node : m_Nodes)
                {
                    distances.push_back(las::distance(point, node.s_Point));
                    result.push_back(node.s_Point);
                }
                COrderings::simultaneousSort(distances, result);
            }
        }

        //! Get an iterator over the points in the tree.
        TPointIterator begin(void) const
        {
            return TPointIterator(m_Nodes.begin());
        }

        //! Get an iterator to the end of the points in the tree.
        TPointIterator end(void) const
        {
            return TPointIterator(m_Nodes.end());
        }

        //! A pre-order depth first traversal of the k-d tree nodes.
        //!
        //! \param[in] f The function to apply to the nodes.
        //! \tparam F should have the signature bool (const SNode &).
        //! Traversal stops below point that \p f returns false.
        template<typename F>
        void preorderDepthFirst(F f) const
        {
            if (m_Nodes.empty())
            {
                return;
            }
            this->preorderDepthFirst(m_Nodes[0], f);
        }

        //! A post-order depth first traversal of the k-d tree nodes.
        //!
        //! \param[in] f The function to apply to the nodes.
        //! \tparam F should have the signature void (const SNode &).
        template<typename F>
        void postorderDepthFirst(F f) const
        {
            if (m_Nodes.empty())
            {
                return;
            }
            this->postorderDepthFirst(m_Nodes[0], f);
        }

        //! Check the tree invariants.
        bool checkInvariants(void) const
        {
            for (const auto &node : m_Nodes)
            {
                if (!node.checkInvariants(m_Dimension))
                {
                    return false;
                }
            }
            return true;
        }

    private:
        //! Append a node moving \p point into place.
        void append(True /*move*/, SNode *parent, POINT &point)
        {
            m_Nodes.emplace_back(parent, std::move(point));
        }

        //! Append a node coping \p point into place.
        void append(False /*move*/, SNode *parent, const POINT &point)
        {
            m_Nodes.emplace_back(parent, point);
        }

        //! Recursively build the k-d tree.
        template<typename ITR, typename MOVE>
        SNode *buildRecursively(SNode *parent,
                                std::size_t coordinate,
                                ITR begin, ITR end, MOVE move)
        {
            std::size_t n{static_cast<std::size_t>(end - begin) / 2};
            ITR median{begin + n};
            std::nth_element(begin, median, end, CCoordinateLess(coordinate));
            this->append(move, parent, *median);
            SNode *node{&m_Nodes.back()};
            if (median - begin > 0)
            {
                std::size_t next{(coordinate + 1) % m_Dimension};
                SNode *leftChild{this->buildRecursively(node, next, begin, median, move)};
                node->s_LeftChild = leftChild;
            }
            if (end - median > 1)
            {
                std::size_t next{(coordinate + 1) % m_Dimension};
                SNode *rightChild{this->buildRecursively(node, next, median + 1, end, move)};
                node->s_RightChild = rightChild;
            }
            return node;
        }

        //! Recursively find the nearest point to \p point.
        const POINT *nearestNeighbour(const POINT &point,
                                      const SNode &node,
                                      std::size_t coordinate,
                                      const POINT *nearest,
                                      TCoordinatePrecise &distanceToNearest) const
        {
            TCoordinatePrecise distance{las::distance(point, node.s_Point)};

            if (distance < distanceToNearest)
            {
                nearest = &node.s_Point;
                distanceToNearest = distance;
            }

            if (node.s_LeftChild || node.s_RightChild)
            {
                TCoordinatePrecise distanceToHyperplane{  point(coordinate)
                                                        - node.s_Point(coordinate)};

                SNode *primary{node.s_LeftChild};
                SNode *secondary{node.s_RightChild};
                if (!primary || (secondary && distanceToHyperplane > 0))
                {
                    std::swap(primary, secondary);
                }

                std::size_t nextCoordinate{(coordinate + 1) % m_Dimension};
                nearest = this->nearestNeighbour(point,
                                                 *primary,
                                                 nextCoordinate,
                                                 nearest,
                                                 distanceToNearest);
                if (secondary && std::fabs(distanceToHyperplane) < distanceToNearest)
                {
                    nearest = this->nearestNeighbour(point,
                                                     *secondary,
                                                     nextCoordinate,
                                                     nearest,
                                                     distanceToNearest);
                }
            }

            return nearest;
        }

        //! Recursively find the nearest point to \p point.
        void nearestNeighbours(const POINT &point,
                               const SNode &node,
                               std::size_t coordinate,
                               TNearestAccumulator &nearest) const
        {
            TCoordinatePrecise distance = las::distance(point, node.s_Point);

            nearest.add({distance, node.s_Point});

            if (node.s_LeftChild || node.s_RightChild)
            {
                TCoordinatePrecise distanceToHyperplane =  point(coordinate)
                                                         - node.s_Point(coordinate);

                SNode *primary   = node.s_LeftChild;
                SNode *secondary = node.s_RightChild;
                if (!primary || (secondary && distanceToHyperplane > 0))
                {
                    std::swap(primary, secondary);
                }

                std::size_t nextCoordinate = (coordinate + 1) % m_Dimension;
                this->nearestNeighbours(point, *primary, nextCoordinate, nearest);
                if (secondary && std::fabs(distanceToHyperplane) < nearest.biggest().first)
                {
                    this->nearestNeighbours(point, *secondary, nextCoordinate, nearest);
                }
            }
        }

        //! Visit the branch rooted at \p node with \p f in pre-order.
        template<typename F>
        static void preorderDepthFirst(const SNode &node, F f)
        {
            if (f(node))
            {
                if (node.s_LeftChild)
                {
                    preorderDepthFirst(*node.s_LeftChild, f);
                }
                if (node.s_RightChild)
                {
                    preorderDepthFirst(*node.s_RightChild, f);
                }
            }
        }

        //! Visit the branch rooted at \p node with \p f in post-order.
        template<typename F>
        static void postorderDepthFirst(const SNode &node, F f)
        {
            if (node.s_LeftChild)
            {
                postorderDepthFirst(*node.s_LeftChild, f);
            }
            if (node.s_RightChild)
            {
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
