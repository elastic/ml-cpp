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

#ifndef INCLUDED_ml_maths_CXMeans_h
#define INCLUDED_ml_maths_CXMeans_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/CChecksum.h>
#include <maths/CInformationCriteria.h>
#include <maths/CKMeansFast.h>
#include <maths/CLinearAlgebra.h>
#include <maths/COrderings.h>

#include <boost/unordered_set.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include <stdint.h>


namespace ml {
namespace maths {

//! \brief Implementation of x-means algorithm.
//!
//! DESCRIPTION:\n
//! Implements the x-means algorithm proposed by Pelleg and Moore
//! with a different scoring criterion used during Improve-Structure.
//! See https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf for more
//! details.
//!
//! IMPLEMENTATION DECISIONS:\n
//! The point type is a template parameter for greater flexibility.
//! It must support addition, subtraction, have free functions for
//! coordinate-wise min and max, satisfy the constraints imposed by
//! CBasicStatistics::SSampleCentralMoments, support coordinate access
//! by the brackets operator and have member functions called dimension
//! and euclidean - which gives the Euclidean norm of the vector.
template<typename POINT, typename COST = CSphericalGaussianInfoCriterion<POINT, E_BIC> >
class CXMeans {
    public:
        typedef std::vector<double> TDoubleVec;
        typedef std::vector<POINT> TPointVec;
        typedef std::vector<TPointVec> TPointVecVec;
        typedef boost::unordered_set<uint64_t> TUInt64USet;
        typedef TUInt64USet::iterator TUInt64USetItr;
        typedef typename CBasicStatistics::SSampleMean<POINT>::TAccumulator TMeanAccumulator;

        //! A cluster.
        //!
        //! DESCRIPTION:\n
        //! This associates the cost, cluster centre and points. It
        //! also provides fast comparison by a checksum and sorts the
        //! points for stable comparison.
        class CCluster {
            public:
                CCluster(void) :
                    m_Cost(std::numeric_limits<double>::max()), m_Checksum(0)
                {}

                //! Check for equality using checksum and then points if the
                //! checksum is ambiguous.
                bool operator==(const CCluster &other) const {
                    return m_Checksum == other.m_Checksum && m_Points == other.m_Points;
                }

                //! Total ordering by checksum breaking ties using expensive
                //! comparison on all points.
                bool operator<(const CCluster &rhs) const {
                    return COrderings::lexicographical_compare(m_Checksum, m_Points,
                                                               rhs.m_Checksum, rhs.m_Points);
                }

                //! Get the number of points in the cluster.
                std::size_t size(void) const {
                    return m_Points.size();
                }

                //! Set the cluster cost.
                void cost(double cost) {
                    m_Cost = cost;
                }
                //! Get the cluster cost.
                double cost(void) const {
                    return m_Cost;
                }

                //! Set the cluster centre.
                void centre(const POINT &centre) {
                    m_Centre = centre;
                }
                //! Get the cluster centre.
                const POINT &centre(void) const {
                    return m_Centre;
                }

                //! Swap the points into place and recalculate the checksum.
                void points(TPointVec &points) {
                    m_Points.swap(points);
                    std::sort(m_Points.begin(), m_Points.end());
                    m_Checksum = CChecksum::calculate(0, m_Points);
                }
                //! Get the cluster points.
                const TPointVec &points(void) const {
                    return m_Points;
                }

                //! Get the cluster checksum.
                uint64_t checksum(void) const {
                    return m_Checksum;
                }

            private:
                //! The information criterion cost of this cluster.
                double m_Cost;
                //! The centroid of the points in this cluster.
                POINT m_Centre;
                //! The points in the cluster.
                TPointVec m_Points;
                //! A checksum for the points in the cluster.
                uint64_t m_Checksum;
        };

        typedef std::vector<CCluster> TClusterVec;

    public:
        CXMeans(std::size_t kmax) :
            m_Kmax(kmax),
            m_MinCost(std::numeric_limits<double>::max()) {
            m_BestCentres.reserve(m_Kmax);
            m_Clusters.reserve(m_Kmax);
        }

        //! Set the points to cluster.
        //!
        //! \note These are swapped in to place.
        void setPoints(TPointVec &points) {
            m_Kmeans.setPoints(points);
            m_Clusters.clear();
            m_Clusters.push_back(CCluster());
            double cost = COST(points).calculate();
            m_Clusters[0].cost(cost);
            TMeanAccumulator centroid;
            centroid.add(points);
            m_Clusters[0].centre(CBasicStatistics::mean(centroid));
            m_Clusters[0].points(points);
            m_MinCost = cost;
            m_BestCentres.push_back(CBasicStatistics::mean(centroid));
        }

        //! Get the best centres found to date.
        const TPointVec &centres(void) const {
            return m_BestCentres;
        }

        //! Get the best clusters found to date.
        const TClusterVec &clusters(void) const {
            return m_Clusters;
        }

        //! Run the full x-means algorithm.
        //!
        //! This iterates between improve structure and improve
        //! parameters until either kmax clusters have been created
        //! or there was an improve structure round with no change.
        //!
        //! \param[in] improveParamsKmeansIterations The number of
        //! iterations of Lloyd's algorithm to use in k-means for a
        //! single round of improve parameters.
        //! \param[in] improveStructureClusterSeeds The number of
        //! random seeds to try when initializing k-means for a
        //! single round of improve structure.
        //! \param[in] improveStructureKmeansIterations The number
        //! of iterations of Lloyd's algorithm to use in k-means for
        //! a single round of improve structure.
        void run(std::size_t improveParamsKmeansIterations,
                 std::size_t improveStructureClusterSeeds,
                 std::size_t improveStructureKmeansIterations) {
            while (this->improveStructure(improveStructureClusterSeeds,
                                          improveStructureKmeansIterations)) {
                this->improveParams(improveParamsKmeansIterations);
            }
            this->polish(10 * improveParamsKmeansIterations);
        }

    protected:
        //! Single round of k-means on the full point set with the
        //! current clusters using at most \p kmeansIterations.
        //!
        //! \param[in] kmeansIterations The limit on the number of
        //! iterations of Lloyd's algorithm to use.
        void improveParams(std::size_t kmeansIterations) {
            typedef const CCluster *TClusterCPtr;
            typedef std::vector<TClusterCPtr> TClusterCPtrVec;

            std::size_t n = m_Clusters.size();

            // Setup k-means to run on the current centres and create
            // sorted lookup of the current clusters.
            TPointVec oldCentres;
            oldCentres.reserve(n);
            TClusterCPtrVec oldClusters;
            oldClusters.reserve(n);
            for (std::size_t i = 0u; i < n; ++i) {
                oldCentres.push_back(m_Clusters[i].centre());
                oldClusters.push_back(&m_Clusters[i]);
            }
            std::sort(oldClusters.begin(),
                      oldClusters.end(),
                      COrderings::SPtrLess());
            m_Kmeans.setCentres(oldCentres);

            // k-means to improve parameters.
            m_Kmeans.run(kmeansIterations);
            const TPointVec &newCentres = m_Kmeans.centres();
            TPointVecVec newClusterPoints;
            m_Kmeans.clusters(newClusterPoints);

            // Note that oldClusters holds pointers to the current
            // clusters so we can't overwrite them until after the
            // following loop.

            TClusterVec newClusters;
            newClusters.reserve(newCentres.size());
            TUInt64USet preserved;
            COST cost_;

            for (std::size_t i = 0u; i < n; ++i) {
                newClusters.push_back(CCluster());
                CCluster &cluster = newClusters.back();
                cluster.centre(newCentres[i]);
                cluster.points(newClusterPoints[i]);
                typename TClusterCPtrVec::const_iterator j =
                    std::lower_bound(oldClusters.begin(), oldClusters.end(),
                                     &cluster, COrderings::SPtrLess());
                if (j != oldClusters.end() && **j == cluster) {
                    cluster.cost((*j)->cost());
                    preserved.insert(cluster.checksum());
                } else {
                    cluster.cost(COST(cluster.points()).calculate());
                }
                cost_.add(cluster.points());
            }

            // Refresh the clusters and inactive list.
            m_Clusters.swap(newClusters);
            for (TUInt64USetItr i = m_Inactive.begin(); i != m_Inactive.end(); /**/) {
                if (preserved.count(*i) > 0) {
                    ++i;
                } else {
                    i = m_Inactive.erase(i);
                }
            }

            // Refresh the best clustering found so far.
            double cost = cost_.calculate();
            if (cost < m_MinCost) {
                m_BestCentres.clear();
                for (std::size_t i = 0u; i < n; ++i) {
                    m_BestCentres.push_back(m_Clusters[i].centre());
                }
                m_MinCost = cost;
            }
        }

        //! Try splitting each cluster in two and keep only those
        //! splits which improve the overall score.
        //!
        //! \param[in] clusterSeeds The number of different 2-splits
        //! to try per cluster.
        //! \param[in] kmeansIterations The limit on the number of
        //! iterations of Lloyd's algorithm to use for each k-means.
        bool improveStructure(std::size_t clusterSeeds,
                              std::size_t kmeansIterations) {
            if (m_Clusters.empty()) {
                return false;
            }

            // Declared outside the loop to minimize allocations.
            CKMeansFast<POINT> kmeans;
            TPointVec points;
            TPointVecVec clusterPoints;
            TPointVec bestClusterCentres;
            TPointVecVec bestClusterPoints;
            TPointVec seedClusterCentres;

            std::size_t largest = 0;
            for (std::size_t i = 0u; i < m_Clusters.size(); ++i) {
                largest = std::max(largest, m_Clusters[i].size());
            }

            kmeans.reserve(largest);
            points.reserve(largest);
            clusterPoints.reserve(2);
            bestClusterCentres.reserve(2);
            bestClusterPoints.reserve(2);
            seedClusterCentres.reserve(2);

            bool split = false;

            for (std::size_t i = 0u, n = m_Clusters.size();
                 i < n && m_Clusters.size() < m_Kmax;
                 ++i) {
                if (m_Inactive.count(m_Clusters[i].checksum()) > 0) {
                    continue;
                }

                LOG_TRACE("Working on cluster at " << m_Clusters[i].centre());
                LOG_TRACE("Cluster cost = " << m_Clusters[i].cost());

                points.reserve(m_Clusters[i].size());
                points.assign(m_Clusters[i].points().begin(),
                              m_Clusters[i].points().end());
                kmeans.setPoints(points);
                double minCost = std::numeric_limits<double>::max();

                for (std::size_t j = 0u; j < clusterSeeds; ++j) {
                    this->generateSeedCentres(points, 2, seedClusterCentres);
                    LOG_TRACE("seed centres = "
                              << core::CContainerPrinter::print(seedClusterCentres));

                    kmeans.setCentres(seedClusterCentres);
                    kmeans.run(kmeansIterations);

                    const TPointVec &centres = kmeans.centres();
                    LOG_TRACE("centres = " << core::CContainerPrinter::print(centres));
                    clusterPoints.clear();
                    kmeans.clusters(clusterPoints);

                    double cost = COST(clusterPoints).calculate();
                    LOG_TRACE("cost = " << cost);

                    if (cost < minCost) {
                        minCost = cost;
                        bestClusterCentres.assign(centres.begin(), centres.end());
                        bestClusterPoints.swap(clusterPoints);
                    }
                }

                // Check if we should split.
                if (minCost < m_Clusters[i].cost()) {
                    m_Inactive.erase(m_Clusters[i].checksum());
                    m_Clusters[i].cost(COST(bestClusterPoints[0]).calculate());
                    m_Clusters[i].centre(bestClusterCentres[0]);
                    m_Clusters[i].points(bestClusterPoints[0]);
                    for (std::size_t j = 1u; j < bestClusterCentres.size(); ++j) {
                        m_Clusters.push_back(CCluster());
                        m_Clusters.back().cost(COST(bestClusterPoints[j]).calculate());
                        m_Clusters.back().centre(bestClusterCentres[j]);
                        m_Clusters.back().points(bestClusterPoints[j]);
                    }
                    split = true;
                } else {
                    LOG_TRACE("Setting inactive = " << m_Clusters[i].checksum());
                    m_Inactive.insert(m_Clusters[i].checksum());
                }
            }

            return split;
        }

        //! Get the checksums of the clusters which are inactive.
        const TUInt64USet &inactive(void) const {
            return m_Inactive;
        }

    private:
        //! Generate seed points for the cluster centres in k-splits
        //! of \p points.
        //!
        //! These are used to initialize the k-means centres in the
        //! step to improve structure.
        void generateSeedCentres(const TPointVec &points,
                                 std::size_t k,
                                 TPointVec &result) const {
            CKMeansPlusPlusInitialization<POINT, CPRNG::CXorShift1024Mult> kmeansPlusPlus(m_Rng);
            kmeansPlusPlus.run(points, k, result);
        }

        //! Run k-means to improve the best centres.
        //!
        //! \param[in] kmeansIterations The limit on the number of
        //! iterations of Lloyd's algorithm to use.
        void polish(std::size_t kmeansIterations) {
            if (m_BestCentres.size() > 1) {
                m_Kmeans.setCentres(m_BestCentres);
                m_Kmeans.run(kmeansIterations);
                m_BestCentres = m_Kmeans.centres();
                TPointVecVec polishedClusterPoints;
                m_Kmeans.clusters(polishedClusterPoints);
                m_Clusters.clear();
                m_Clusters.reserve(m_BestCentres.size());
                for (std::size_t i = 0u; i < m_BestCentres.size(); ++i) {
                    m_Clusters.push_back(CCluster());
                    CCluster &cluster = m_Clusters.back();
                    cluster.cost(COST(polishedClusterPoints[i]).calculate());
                    cluster.centre(m_BestCentres[i]);
                    cluster.points(polishedClusterPoints[i]);
                }
            }
        }

    private:
        //! The random number generator.
        mutable CPRNG::CXorShift1024Mult m_Rng;

        //! The maximum number of clusters.
        std::size_t m_Kmax;

        //! The current clusters.
        TClusterVec m_Clusters;

        //! Checksums of clusters which weren't modified in the last
        //! iteration.
        TUInt64USet m_Inactive;

        //! The fast k-means state for the full set of points.
        CKMeansFast<POINT> m_Kmeans;

        //! The minimum cost clustering found to date.
        double m_MinCost;

        //! The cluster centres corresponding to the maximum score.
        TPointVec m_BestCentres;
};

}
}

#endif // INCLUDED_ml_maths_CXMeans_h
