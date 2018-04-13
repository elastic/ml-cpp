/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CKMeansFast_h
#define INCLUDED_ml_maths_CKMeansFast_h

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoundingBox.h>
#include <maths/CChecksum.h>
#include <maths/CKdTree.h>
#include <maths/CSampling.h>
#include <maths/CTypeConversions.h>

#include <boost/iterator/counting_iterator.hpp>

#include <cstddef>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {
namespace kmeans_fast_detail {
using TSizeVec = std::vector<std::size_t>;

//! Get the closest filtered centre to \p point.
template<typename POINT, typename ITR>
std::size_t closest(const std::vector<POINT>& centres, ITR filter, ITR end, const POINT& point) {
    std::size_t result = *filter;
    double d = (point - centres[result]).euclidean();
    for (++filter; filter != end; ++filter) {
        double di = (point - centres[*filter]).euclidean();
        if (di < d) {
            result = *filter;
            d = di;
        }
    }
    return result;
}

//! Get the closest filtered centre to \p point.
template<typename POINT>
std::size_t closest(const std::vector<POINT>& centres, const TSizeVec& filter, const POINT& point) {
    return closest(centres, filter.begin(), filter.end(), point);
}
}

//! \brief Implementation of efficient k-means algorithm.
//!
//! DESCRIPTION:\n
//! Implements the scheme for accelerating k-means proposed by Kanungo
//! et al. This stores the points in a k-d tree and propagates their
//! bounding boxes and centroids up the tree in a post-order depth
//! first traversal of the tree. A branch and bound scheme is used to
//! remove all centres which are definitely further from the bounding
//! box in a pre-order depth first pass over the tree when assigning
//! points to centres in an iteration of Lloyd's algorithm. This can
//! terminate on a given branch once only a single centre remains. See
//! https://www.cs.umd.edu/~mount/Projects/KMeans/pami02.pdf for details.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The point type is a template parameter for greater flexibility.
//! It must support addition, subtraction, have free functions for
//! coordinate-wise min and max, satisfy the constraints imposed by
//! CBasicStatistics::SSampleCentralMoments, support coordinate access
//! by the brackets operator and have member functions called dimension
//! and euclidean - which gives the Euclidean norm of the vector.
template<typename POINT>
class CKMeansFast {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TPointPointPr = std::pair<POINT, POINT>;
    using TPointVec = std::vector<POINT>;
    using TPointVecVec = std::vector<TPointVec>;

    //! A cluster.
    //!
    //! DESCRIPTION:\n
    //! This associates cluster centre and points. It
    //! also provides fast comparison by a checksum and sorts the
    //! points for stable comparison.
    class CCluster {
    public:
        CCluster() : m_Checksum(0) {}

        //! Check for equality using checksum and then points if the
        //! checksum is ambiguous.
        bool operator==(const CCluster& other) const { return m_Checksum == other.m_Checksum && m_Points == other.m_Points; }

        //! Total ordering by checksum breaking ties using expensive
        //! comparison on all points.
        bool operator<(const CCluster& rhs) const {
            return m_Checksum < rhs.m_Checksum || (m_Checksum == rhs.m_Checksum && m_Points < rhs.m_Points);
        }

        //! Get the number of points in the cluster.
        std::size_t size() const { return m_Points.size(); }

        //! Set the cluster centre.
        void centre(const POINT& centre) { m_Centre = centre; }
        //! Get the cluster centre.
        const POINT& centre() const { return m_Centre; }

        //! Swap the points into place and recalculate the checksum.
        void points(TPointVec& points) {
            m_Points.swap(points);
            std::sort(m_Points.begin(), m_Points.end());
            m_Checksum = CChecksum::calculate(0, m_Points);
        }
        //! Get the cluster points.
        const TPointVec& points() const { return m_Points; }

        //! Get the cluster checksum.
        uint64_t checksum() const { return m_Checksum; }

    private:
        //! The centroid of the points in this cluster.
        POINT m_Centre;
        //! The points in the cluster.
        TPointVec m_Points;
        //! A checksum for the points in the cluster.
        uint64_t m_Checksum;
    };

    using TClusterVec = std::vector<CCluster>;

protected:
    using TBarePoint = typename SStripped<POINT>::Type;
    using TBarePointPrecise = typename SFloatingPoint<TBarePoint, double>::Type;
    using TMeanAccumulator = typename CBasicStatistics::SSampleMean<TBarePointPrecise>::TAccumulator;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TBoundingBox = CBoundingBox<TBarePoint>;
    class CKdTreeNodeData;
    using TNode = typename CKdTree<POINT, CKdTreeNodeData>::SNode;

    //! \brief The data the x-means algorithm needs at each k-d
    //! tree node.
    //!
    //! DESCRIPTION:\n
    //! At every node the algorithm needs the axis aligned
    //! bounding box of the points in the branch rooted at that
    //! node together with their centroid.
    class CKdTreeNodeData {
    public:
        CKdTreeNodeData() {}
        explicit CKdTreeNodeData(const POINT& x) : m_BoundingBox(x), m_Centroid() { m_Centroid.add(x); }

        //! Get the bounding box.
        const TBoundingBox& boundingBox() const { return m_BoundingBox; }

        //! Get the centroid.
        const TMeanAccumulator& centroid() const { return m_Centroid; }

        //! Combine the bounding boxes and centroids.
        void add(const CKdTreeNodeData& other) const {
            m_BoundingBox.add(other.m_BoundingBox);
            m_Centroid += other.m_Centroid;
        }

        //! Add \p x to the bounding box and centroid.
        void add(const POINT& x) const {
            m_BoundingBox.add(x);
            m_Centroid.add(x);
        }

        //! Clear the bounding box and centroid.
        void clear() const {
            m_BoundingBox.clear();
            m_Centroid = TMeanAccumulator();
        }

    private:
        //! The points' bounding box.
        mutable TBoundingBox m_BoundingBox;
        //! The centroid of the points.
        mutable TMeanAccumulator m_Centroid;
    };

    //! \brief Propagates data needed by the x-means algorithm
    //! up the k-d tree.
    //!
    //! DESCRIPTION:\n
    //! At every node the algorithm needs the axis aligned
    //! bounding box of the points in the branch together with
    //! their centroid. This can be computed in a single post-
    //! order depth first traversal of the k-d tree. This annotates
    //! the data onto the k-d tree nodes.
    struct SDataPropagator {
        //! Propagate the data to \p node.
        bool operator()(const TNode& node) const {
            node.clear();
            node.add(node.s_Point);
            this->propagate(node.s_LeftChild, node);
            this->propagate(node.s_RightChild, node);
            return true;
        }

        //! Update \p data with the data from \p child.
        void propagate(const TNode* child, const CKdTreeNodeData& data) const {
            if (child) {
                data.add(*child);
            }
        }
    };

    //! \brief Maintains a set of candidate centres which could
    //! be the closest centre to a point in k-d tree branch.
    //!
    //! DESCRIPTION\n
    //! This is responsible for propagating the cluster centres
    //! down the k-d tree. The idea is that cluster centres are
    //! removed when it is determined that they are further from
    //! all points in the branch than some other cluster centre.
    //! See http://www.cs.umd.edu/~mount/Projects/KMeans/pami02.pdf
    //! for more details.
    class CCentreFilter {
    public:
        //! \brief Predicate used to compute whether a centre
        //! is further from the bounding box of a collection
        //! of points than a specified point.
        class CFurtherFrom {
        public:
            CFurtherFrom(const TBoundingBox& bb_, std::size_t x_, const TPointVec& centres_) : bb(&bb_), x(x_), centres(&centres_) {}

            bool operator()(std::size_t y) const { return y == x ? false : bb->closerToX((*centres)[x], (*centres)[y]); }

        private:
            const TBoundingBox* bb;
            std::size_t x;
            const TPointVec* centres;
        };

    public:
        explicit CCentreFilter(const TPointVec& centres)
            : m_Centres(&centres),
              m_Filter(boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(centres.size())) {}

        //! Get the centres.
        const TPointVec& centres() const { return *m_Centres; }

        //! Get the filter.
        const TSizeVec& filter() const { return m_Filter; }

        //! Update the filter with to remove all centres which
        //! are further from \p bb than one of the current centres
        //! in the filter.
        //!
        //! This is the *key* step in the acceleration of k-means.
        //! The idea is to first find the point closest to the
        //! centre of \p bb and then remove all centres which
        //! are further than this from every point in the bounding
        //! box. The farthest point in a bounding box must be one
        //! of the 2^d corners of the bounding box. However, this
        //! can be found in O(d). (See CBoundingBox::closerToX.)
        //!
        //! The centres are propagated down the k-d tree in a pre-
        //! order depth first traversal. As soon as one centre is
        //! closer to the bounding box of the points in a branch
        //! the traversal can terminate and update the centre with
        //! their centroid.
        void prune(const TBoundingBox& bb) {
            namespace detail = kmeans_fast_detail;

            if (m_Filter.size() > 1) {
                std::size_t closest = detail::closest(*m_Centres, m_Filter, POINT(bb.centre()));
                m_Filter.erase(std::remove_if(m_Filter.begin(), m_Filter.end(), CFurtherFrom(bb, closest, *m_Centres)), m_Filter.end());
            }
        }

    private:
        //! The current centres.
        const TPointVec* m_Centres;

        //! The centres which could be closer to one of the points
        //! in the current branch of the k-d tree.
        TSizeVec m_Filter;
    };

    //! \brief Updates the cluster centres in an iteration of Lloyd's
    //! algorithm.
    //!
    //! DESCRIPTION:\n
    //! This is used in a pre-order depth first traversal of the
    //! k-d tree of points to efficiently update the cluster centres
    //! in one iteration of Lloyd's algorithm. Each point is assigned
    //! to its closest centre and the centre placed at the centroid
    //! of its assigned points.
    class CCentroidComputer {
    public:
        CCentroidComputer(const TPointVec& centres, TMeanAccumulatorVec& centroids) : m_Centres(centres), m_Centroids(&centroids) {}

        //! Update the centres with \p node.
        //!
        //! \return True if we need to recurse and false otherwise.
        bool operator()(const TNode& node) {
            namespace detail = kmeans_fast_detail;

            m_Centres.prune(node.boundingBox());
            const TSizeVec& filter = m_Centres.filter();
            if (filter.size() == 1) {
                (*m_Centroids)[filter[0]] += node.centroid();
                return false;
            } else {
                const TPointVec& centres = m_Centres.centres();
                const POINT& point = node.s_Point;
                (*m_Centroids)[detail::closest(centres, filter, point)].add(point);
            }
            return true;
        }

    private:
        //! The current centres.
        CCentreFilter m_Centres;

        //! Compute the new cluster centres.
        TMeanAccumulatorVec* m_Centroids;
    };

    //! \brief Extracts the closest points to each centre from a
    //! k-d tree in a single traversal.
    //!
    //! DESCRIPTION:\n
    //! This is used in a post-order depth first traversal of the
    //! k-d tree of points to extract the closest points to each
    //! centre supplied to the constructor.
    class CClosestPointsCollector {
    public:
        CClosestPointsCollector(std::size_t numberPoints, const TPointVec& centres, TPointVecVec& closestPoints)
            : m_Centres(&centres), m_ClosestPoints(&closestPoints) {
            m_ClosestPoints->resize(centres.size());
            for (std::size_t i = 0u; i < m_ClosestPoints->size(); ++i) {
                (*m_ClosestPoints)[i].clear();
                (*m_ClosestPoints)[i].reserve(numberPoints / m_ClosestPoints->size() + 1);
            }
        }

        //! Add \p node's point to the closest centre's nearest
        //! point collection.
        void operator()(const TNode& node) {
            namespace detail = kmeans_fast_detail;
            std::size_t n = m_Centres->size();
            const POINT& point = node.s_Point;
            (*m_ClosestPoints)[detail::closest(
                                   *m_Centres, boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(n), point)]
                .push_back(point);
        }

    private:
        const TPointVec* m_Centres;
        TPointVecVec* m_ClosestPoints;
    };

public:
    //! Reserve space for \p n points.
    void reserve(std::size_t n) { m_Points.reserve(n); }

    //! Set the points to cluster.
    //!
    //! \note \p points are reordered by this operation.
    bool setPoints(TPointVec& points) {
        m_Points.build(points);
        try {
            m_Points.postorderDepthFirst(SDataPropagator());
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to set up k-d tree state: " << e.what());
            return false;
        }
        return true;
    }

    //! Set the initial centres to use.
    //!
    //! \note These are swapped in to place.
    void setCentres(TPointVec& centres) { m_Centres.swap(centres); }

    //! A run of the k-means algorithm using at most \p maxIterations
    //! of Lloyd's algorithm.
    //!
    //! \return True if it converged and false otherwise.
    bool run(std::size_t maxIterations) {
        if (m_Centres.empty()) {
            return true;
        }
        for (std::size_t i = 0u; i < maxIterations; ++i) {
            if (!this->updateCentres()) {
                return true;
            }
        }
        return false;
    }

    //! Get the clusters.
    //!
    //! \param[out] result Filled in with the k clusters.
    void clusters(TClusterVec& result) const {
        result.clear();
        if (m_Centres.empty()) {
            return;
        }
        result.resize(m_Centres.size());
        TPointVecVec clusters;
        this->clusters(clusters);
        for (std::size_t i = 0u; i < m_Centres.size(); ++i) {
            result[i].centre(m_Centres[i]);
            result[i].points(clusters[i]);
        }
    }

    //! Get the points in each cluster.
    //!
    //! \param[out] result Filled in with the closest point to each
    //! of the k centres.
    void clusters(TPointVecVec& result) const {
        result.clear();
        if (m_Centres.empty()) {
            return;
        }
        CClosestPointsCollector collector(m_Points.size(), m_Centres, result);
        m_Points.postorderDepthFirst(collector);
    }

    //! Get the cluster centres.
    const TPointVec& centres() const { return m_Centres; }

private:
    //! Single iteration of Lloyd's algorithm to update \p centres.
    bool updateCentres() {
        using TCoordinate = typename SCoordinate<POINT>::Type;
        static const TCoordinate PRECISION = TCoordinate(5) * std::numeric_limits<TCoordinate>::epsilon();
        TMeanAccumulatorVec newCentres(m_Centres.size());
        CCentroidComputer computer(m_Centres, newCentres);
        m_Points.preorderDepthFirst(computer);
        bool changed = false;
        for (std::size_t i = 0u; i < newCentres.size(); ++i) {
            POINT newCentre(CBasicStatistics::mean(newCentres[i]));
            if ((m_Centres[i] - newCentre).euclidean() > PRECISION * m_Centres[i].euclidean()) {
                m_Centres[i] = newCentre;
                changed = true;
            }
        }
        return changed;
    }

private:
    //! The current cluster centroids.
    TPointVec m_Centres;

    //! The points.
    CKdTree<POINT, CKdTreeNodeData> m_Points;
};

//! \brief Implements "Arthur and Vassilvitskii"'s seed scheme for
//! initializing the centres for k-means.
//!
//! DESCRIPTION:\n
//! See https://en.wikipedia.org/wiki/K-means%2B%2B for details.
template<typename POINT, typename RNG>
class CKMeansPlusPlusInitialization : private core::CNonCopyable {
public:
    using TDoubleVec = std::vector<double>;
    using TSizeVec = std::vector<std::size_t>;
    using TPointVec = std::vector<POINT>;

public:
    CKMeansPlusPlusInitialization(RNG& rng) : m_Rng(rng) {}

    //! Run the k-means++ centre selection algorithm on \p points.
    //!
    //! \param[in] points The points to cluster.
    //! \param[in] k The number of seed centres to generate.
    //! \param[out] result Filled in with the seed centres.
    void run(const TPointVec& points, std::size_t k, TPointVec& result) const {
        result.clear();
        if (points.empty() || k == 0) {
            return;
        }

        result.reserve(k);

        std::size_t n = points.size();
        LOG_TRACE(<< "# points = " << n);

        TSizeVec centre;
        CSampling::uniformSample(m_Rng, 0, n, 1, centre);
        LOG_TRACE(<< "centre = " << centre[0]);

        result.push_back(points[centre[0]]);
        LOG_TRACE(<< "centres to date = " << core::CContainerPrinter::print(result));

        TDoubleVec distances;
        TPointVec centres_;
        CKdTree<POINT> centres;
        distances.resize(n);
        centres_.reserve(k);
        centres.reserve(k);

        for (std::size_t i = 1u; i < k; ++i) {
            centres_.assign(result.begin(), result.end());
            centres.build(centres_);

            for (std::size_t j = 0u; j < n; ++j) {
                const POINT* nn = centres.nearestNeighbour(points[j]);
                distances[j] = nn ? square((points[j] - *nn).euclidean()) : 0.0;
            }

            centre[0] = CSampling::categoricalSample(m_Rng, distances);
            LOG_TRACE(<< "centre = " << centre[0]);

            result.push_back(points[centre[0]]);
            LOG_TRACE(<< "centres to date = " << core::CContainerPrinter::print(result));
        }
    }

private:
    //! Compute \p x square.
    double square(double x) const { return x * x; }

private:
    //! The random number generator.
    RNG& m_Rng;
};
}
}

#endif // INCLUDED_ml_maths_CKMeansFast_h
