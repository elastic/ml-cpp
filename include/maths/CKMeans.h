/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CKMeans_h
#define INCLUDED_ml_maths_CKMeans_h

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CBoundingBox.h>
#include <maths/CChecksum.h>
#include <maths/CKdTree.h>
#include <maths/CLinearAlgebraShims.h>
#include <maths/CSampling.h>
#include <maths/CTools.h>
#include <maths/CTypeTraits.h>

#include <boost/iterator/counting_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

namespace ml {
namespace maths {

//! \brief Implementation of the k-means algorithm.
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
//! It must support addition, subtraction, satisfy the constraints
//! imposed by CBasicStatistics::SSampleCentralMoments and support
//! coordinate access by the brackets operator. Coordinatewise min
//! and max, the dimension and the Euclidean distance between a pair
//! of points are handled by overloading the appropriate functions in
//! the las namespace.
template<typename POINT>
class CKMeans {
public:
    using TSizeVec = std::vector<std::size_t>;
    using TPointPointPr = std::pair<POINT, POINT>;
    using TPointVec = std::vector<POINT>;
    using TPointVecVec = std::vector<TPointVec>;

protected:
    class CKdTreeNodeData;

public:
    using TKdTree = CKdTree<POINT, CKdTreeNodeData>;
    using TPointCItr = typename TKdTree::TPointCItr;

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
        bool operator==(const CCluster& other) const {
            return m_Checksum == other.m_Checksum && m_Points == other.m_Points;
        }

        //! Total ordering by checksum breaking ties using expensive
        //! comparison on all points.
        bool operator<(const CCluster& rhs) const {
            return m_Checksum < rhs.m_Checksum ||
                   (m_Checksum == rhs.m_Checksum && m_Points < rhs.m_Points);
        }

        //! Get the number of points in the cluster.
        std::size_t size() const { return m_Points.size(); }

        //! Set the cluster centre.
        void centre(const POINT& centre) { m_Centre = centre; }
        //! Get the cluster centre.
        const POINT& centre() const { return m_Centre; }

        //! Swap the points into place and recalculate the checksum.
        void points(TPointVec points) {
            m_Points = std::move(points);
            std::sort(m_Points.begin(), m_Points.end());
            m_Checksum = CChecksum::calculate(0, m_Points);
        }
        //! Get the cluster points.
        const TPointVec& points() const { return m_Points; }

        //! Get the cluster checksum.
        std::uint64_t checksum() const { return m_Checksum; }

    private:
        //! The centroid of the points in this cluster.
        POINT m_Centre;
        //! The points in the cluster.
        TPointVec m_Points;
        //! A checksum for the points in the cluster.
        std::uint64_t m_Checksum;
    };

    using TClusterVec = std::vector<CCluster>;

public:
    //! Compute the mean dispersion between \p clusters and \p centres.
    //!
    //! \note IMPORTANT This assumes there is a one-to-one mapping from
    //! \p centres to the collections of points in \p clusters.
    static double dispersion(const TPointVec& centres, const TPointVecVec& clusters) {
        CBasicStatistics::SSampleMean<double>::TAccumulator result;
        for (std::size_t i = 0u; i < centres.size(); ++i) {
            for (const auto& point : clusters[i]) {
                result.add(las::distance(centres[i], point));
            }
        }
        return CBasicStatistics::mean(result);
    }

    //! Reserve space for \p n points.
    void reserve(std::size_t n) { m_Points.reserve(n); }

    //! Set the points to cluster.
    //!
    //! \note \p points are reordered by this operation.
    bool setPoints(TPointVec points) {
        m_Points.build(std::move(points));
        try {
            m_Points.postorderDepthFirst(SDataPropagator());
        } catch (const std::exception& e) {
            LOG_ERROR(<< "Failed to set up k-d tree state: " << e.what());
            return false;
        }
        return true;
    }

    //! Get an iterator over the points to cluster.
    TPointCItr beginPoints() const { return m_Points.begin(); }

    //! Get an iterator to the end of the points to cluster.
    TPointCItr endPoints() const { return m_Points.end(); }

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
        TMeanAccumulatorVec newCentres;
        for (std::size_t i = 0u; i < maxIterations; ++i) {
            if (!this->updateCentres(newCentres)) {
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
            result[i].points(std::move(clusters[i]));
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

protected:
    using TCoordinate = typename SCoordinate<POINT>::Type;
    using TBarePoint = typename SStripped<POINT>::Type;
    using TBarePointPrecise = typename SFloatingPoint<TBarePoint, double>::Type;
    using TMeanAccumulator =
        typename CBasicStatistics::SSampleMean<TBarePointPrecise>::TAccumulator;
    using TOptionalMeanAccumulator = boost::optional<TMeanAccumulator>;
    using TMeanAccumulatorVec = std::vector<TMeanAccumulator>;
    using TBoundingBox = CBoundingBox<TBarePoint>;
    using TNode = typename TKdTree::SNode;

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
        explicit CKdTreeNodeData(const POINT& x) : m_BoundingBox(x) {
            this->initialize(x);
            m_Centroid->add(x);
        }

        //! Get the bounding box.
        const TBoundingBox& boundingBox() const { return m_BoundingBox; }

        //! Get the centroid.
        const TOptionalMeanAccumulator& centroid() const { return m_Centroid; }

        //! Combine the bounding boxes and centroids.
        void add(const CKdTreeNodeData& other) const {
            m_BoundingBox.add(other.m_BoundingBox);
            if (m_Centroid && other.m_Centroid) {
                *m_Centroid += *other.m_Centroid;
            } else if (other.m_Centroid) {
                m_Centroid = other.m_Centroid;
            }
        }

        //! Add \p x to the bounding box and centroid.
        void add(const POINT& x) const {
            this->initialize(x);
            m_BoundingBox.add(x);
            m_Centroid->add(x);
        }

        //! Clear the bounding box and centroid.
        void clear() const {
            m_BoundingBox.clear();
            m_Centroid.reset();
        }

    private:
        //! Initialize the centroid if it is null.
        void initialize(const POINT& x) const {
            if (!m_Centroid) {
                m_Centroid.reset(TMeanAccumulator(las::zero(x)));
            }
        }

    private:
        //! The points' bounding box.
        mutable TBoundingBox m_BoundingBox;
        //! The centroid of the points.
        mutable TOptionalMeanAccumulator m_Centroid;
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
            CFurtherFrom(const TBoundingBox& bb_, std::size_t x_, const TPointVec& centres_)
                : bb(&bb_), x(x_), centres(&centres_) {}

            bool operator()(std::size_t y) const {
                return y == x ? false : bb->closerToX((*centres)[x], (*centres)[y]);
            }

        private:
            const TBoundingBox* bb;
            std::size_t x;
            const TPointVec* centres;
        };

    public:
        explicit CCentreFilter(const TPointVec& centres)
            : m_Centres(&centres),
              m_Filter(boost::counting_iterator<std::size_t>(0),
                       boost::counting_iterator<std::size_t>(centres.size())) {}

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
            if (m_Filter.size() > 1) {
                std::size_t closest_ = closest(*m_Centres, m_Filter, POINT(bb.centre()));
                m_Filter.erase(std::remove_if(m_Filter.begin(), m_Filter.end(),
                                              CFurtherFrom(bb, closest_, *m_Centres)),
                               m_Filter.end());
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
        CCentroidComputer(const TPointVec& centres, TMeanAccumulatorVec& centroids)
            : m_Centres(centres), m_Centroids(&centroids) {}

        //! Update the centres with \p node.
        //!
        //! \return True if we need to recurse and false otherwise.
        bool operator()(const TNode& node) {
            m_Centres.prune(node.boundingBox());
            const TSizeVec& filter = m_Centres.filter();
            if (filter.size() == 1) {
                if (node.centroid()) {
                    (*m_Centroids)[filter[0]] += *node.centroid();
                }
                return false;
            } else {
                const TPointVec& centres = m_Centres.centres();
                const POINT& point = node.s_Point;
                (*m_Centroids)[closest(centres, filter, point)].add(point);
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
        CClosestPointsCollector(std::size_t numberPoints,
                                const TPointVec& centres,
                                TPointVecVec& closestPoints)
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
            std::size_t n = m_Centres->size();
            const POINT& point = node.s_Point;
            (*m_ClosestPoints)[closest(*m_Centres, boost::counting_iterator<std::size_t>(0),
                                       boost::counting_iterator<std::size_t>(n), point)]
                .push_back(point);
        }

    private:
        const TPointVec* m_Centres;
        TPointVecVec* m_ClosestPoints;
    };

private:
    //! Single iteration of Lloyd's algorithm to update \p centres.
    bool updateCentres(TMeanAccumulatorVec& newCentres) {
        const TCoordinate precision{TCoordinate(5) *
                                    std::numeric_limits<TCoordinate>::epsilon()};
        newCentres.assign(m_Centres.size(), TMeanAccumulator(las::zero(m_Centres[0])));
        CCentroidComputer computer(m_Centres, newCentres);
        m_Points.preorderDepthFirst(computer);
        bool changed = false;
        POINT newCentre;
        for (std::size_t i = 0u; i < newCentres.size(); ++i) {
            newCentre = CBasicStatistics::mean(newCentres[i]);
            if (las::distance(m_Centres[i], newCentre) >
                precision * las::norm(m_Centres[i])) {
                using std::swap;
                swap(m_Centres[i], newCentre);
                changed = true;
            }
        }
        return changed;
    }

    //! Get the closest filtered centre to \p point.
    template<typename ITR>
    static std::size_t
    closest(const TPointVec& centres, ITR filter, ITR end, const POINT& point) {
        std::size_t result = *filter;
        double d = las::distance(point, centres[result]);
        for (++filter; filter != end; ++filter) {
            double di = las::distance(point, centres[*filter]);
            if (di < d) {
                result = *filter;
                d = di;
            }
        }
        return result;
    }

    //! Get the closest filtered centre to \p point.
    static std::size_t
    closest(const TPointVec& centres, const TSizeVec& filter, const POINT& point) {
        return closest(centres, filter.begin(), filter.end(), point);
    }

private:
    //! The current cluster centroids.
    TPointVec m_Centres;

    //! The points.
    CKdTree<POINT, CKdTreeNodeData> m_Points;
};

//! \brief Implements "Arthur and Vassilvitskii"'s seed scheme for initializing
//! the centres for k-means.
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
    //! Calls run on [ \p points.begin(), \p points.end() ).
    void run(const TPointVec& points, std::size_t k, TPointVec& result) const {
        run(points.begin(), points.end(), k, result);
    }

    //! Run the k-means++ centre selection algorithm on [ \p beginPoints, \p endPoints ).
    //!
    //! \param[in] beginPoints The first point to cluster.
    //! \param[in] endPoints The end of the points to cluster.
    //! \param[in] k The number of seed centres to generate.
    //! \param[out] result Filled in with the seed centres.
    template<typename ITR>
    void run(ITR beginPoints, ITR endPoints, std::size_t k, TPointVec& result) const {
        result.clear();
        if (beginPoints == endPoints || k == 0) {
            return;
        }

        std::size_t n = std::distance(beginPoints, endPoints);
        LOG_TRACE(<< "# points = " << n);

        std::size_t select = CSampling::uniformSample(m_Rng, std::size_t(0), n);
        LOG_TRACE(<< "select = " << select);

        result.reserve(k);
        result.push_back(beginPoints[select]);
        LOG_TRACE(<< "selected to date = " << core::CContainerPrinter::print(result));

        TDoubleVec distances(n, 0.0);
        CKdTree<POINT> selected;
        selected.reserve(k);

        for (std::size_t i = 1; i < k; ++i) {

            selected.build(result);

            std::size_t j{0};
            for (ITR point = beginPoints; point != endPoints; ++j, ++point) {
                const POINT* nn = selected.nearestNeighbour(*point);
                distances[j] = nn != nullptr ? CTools::pow2(las::distance(*point, *nn)) : 0.0;
            }

            select = CSampling::categoricalSample(m_Rng, distances);
            LOG_TRACE(<< "select = " << select);

            result.push_back(beginPoints[select]);
            LOG_TRACE(<< "selected to date = " << core::CContainerPrinter::print(result));
        }
    }

private:
    //! The random number generator.
    RNG& m_Rng;
};
}
}

#endif // INCLUDED_ml_maths_CKMeans_h
