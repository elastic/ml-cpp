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

#ifndef INCLUDED_ml_maths_CXMeansOnline1d_h
#define INCLUDED_ml_maths_CXMeansOnline1d_h

#include <core/CMemory.h>

#include <maths/CClusterer.h>
#include <maths/CNaturalBreaksClassifier.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/Constants.h>
#include <maths/ImportExport.h>

#include <boost/optional.hpp>

#include <string>
#include <utility>
#include <vector>

class CXMeansOnline1dTest;

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief Encodes the distributions available to model the modes.
class MATHS_EXPORT CAvailableModeDistributions {
public:
    static const int NORMAL = 1;
    static const int GAMMA = 2;
    static const int LOG_NORMAL = 4;
    static const int ALL = NORMAL + GAMMA + LOG_NORMAL;

    CAvailableModeDistributions(int value);

    //! Add the available distributions from \p rhs.
    const CAvailableModeDistributions& operator+(const CAvailableModeDistributions& rhs);

    //! Get the number of parameters used to model a mode.
    double parameters() const;

    //! Check if the normal distribution is available.
    bool haveNormal() const;
    //! Check if the gamma distribution is available.
    bool haveGamma() const;
    //! Check if the log-normal distribution is available.
    bool haveLogNormal() const;

    //! Conversion to a string.
    std::string toString() const;
    //! Set from a string.
    bool fromString(const std::string& value);

private:
    //! The encoding.
    int m_Value;
};

//! \brief A single pass online clusterer based on the x-means
//! algorithm of Pelleg and Moore.
//!
//! DESCRIPTION:\n
//! This implements the CClusterer contract targeting the k-means
//! optimization objective (the standard implementation is not possible
//! because the data are processed using the turnstile model).
//!
//! As with x-means clustering proceeds top down, so we only create
//! a split when we are confident. This minimizes our state size and
//! is important for anomaly detection because over splitting the data
//! significantly impairs anomaly detection. An information theoretic
//! criterion is used to decide whether or not to split clusters;
//! specifically, we compare the BIC of the most promising split with
//! that of the unsplit data and require strong evidence for the split.
//! A sketch data structure is used to keep state size down. This
//! allows us to find the approximate minimum within cluster variation
//! split of the full data, the optimal split for the sketch being
//! found by dynamic program, (see CNaturalBreaksClassifier for more
//! detail). In addition, clusters can be merged if they are no longer
//! worth keeping (again based on BIC, but with some hysteresis). This
//! is achieved by aging clusters' prior distributions back to non-
//! informative over time.
//!
//! Verses standard x-means this has one additional feature which is
//! particularly desirable for anomaly detection: the minimum cluster
//! size (in number of data points) and fraction (in proportion of data
//! points) can both be controlled. Typically we don't want to create
//! clusters for a small number of well separated points, since these
//! may be genuine anomalies. This principally affects the search for
//! candidate splits of the data.
//!
//! Note that this is a soft clustering so that we assign the soft
//! membership of a point to a cluster based on the probability that
//! it is generated by the corresponding normal. However, this is not
//! trying to learn a mixture distribution representation of the data,
//! which would require more involved inference scheme such as EM. It
//! is expected to give largely order (of points processed) invariant
//! unsupervised clustering of the data which identifies reasonably
//! well separated clusters.
class MATHS_EXPORT CXMeansOnline1d : public CClusterer1d {
public:
    class CCluster;
    using TDoubleVec = CClusterer1d::TPointPreciseVec;
    using TDoubleDoublePrVec = CClusterer1d::TPointPreciseDoublePrVec;
    using TClusterClusterPr = std::pair<CCluster, CCluster>;
    using TOptionalClusterClusterPr = boost::optional<TClusterClusterPr>;
    using TDoubleDoublePr = std::pair<double, double>;
    using CClusterer1d::add;

    //! \brief Represents a cluster.
    class MATHS_EXPORT CCluster {
    public:
        explicit CCluster(const CXMeansOnline1d& clusterer);

        //! Construct by traversing a state document
        bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                    core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to the supplied inserter
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Set the type of data in the cluster.
        void dataType(maths_t::EDataType dataType);

        //! Add \p point to this cluster.
        void add(double point, double count);

        //! Set the rate at which information is aged out.
        void decayRate(double decayRate);

        //! Propagate the cluster forwards by \p time.
        void propagateForwardsByTime(double time);

        //! Get the unique index of this cluster.
        std::size_t index() const;

        //! Get the "centroid" of the cluster. This is the mean of the prior.
        double centre() const;

        //! Get the "spread" of the cluster. This is variance of the prior.
        double spread() const;

        //! Get the count \p p percentile position within the cluster.
        double percentile(double p) const;

        //! Get the total count of values added to the cluster.
        double count() const;

        //! Get the weight of the cluster.
        double weight(maths_t::EClusterWeightCalc calc) const;

        //! Get the likelihood that \p point is from this cluster.
        double logLikelihoodFromCluster(maths_t::EClusterWeightCalc calc, double point) const;

        //! Get \p numberSamples from this cluster.
        void sample(std::size_t numberSamples, double smallest, double largest, TDoubleVec& samples) const;

        //! Try and find a split by a full search of the binary tree
        //! of possible optimal 2-splits of the data.
        //!
        //! \param[in] distributions The distributions available to
        //! model the clusters.
        //! \param[in] minimumCount The minimum count of a cluster
        //! in the split.
        //! \param[in] smallest The smallest sample added to date.
        //! \param[in] interval The Winsorisation interval.
        //! \param[in] indexGenerator The unique cluster identifier
        //! generator.
        TOptionalClusterClusterPr split(CAvailableModeDistributions distributions,
                                        double minimumCount,
                                        double smallest,
                                        const TDoubleDoublePr& interval,
                                        CIndexGenerator& indexGenerator);

        //! Check if this and \p other cluster should merge.
        //!
        //! \param[in] other The cluster to merge with this one.
        //! \param[in] distributions The distributions available to
        //! model the clusters.
        //! \param[in] smallest The smallest sample added to date.
        //! \param[in] interval The Winsorisation interval.
        bool shouldMerge(CCluster& other,
                         CAvailableModeDistributions distributions,
                         double smallest,
                         const TDoubleDoublePr& interval);

        //! Merge this and \p other cluster.
        CCluster merge(CCluster& other, CIndexGenerator& indexGenerator);

        //! Get the prior describing this object.
        const CNormalMeanPrecConjugate& prior() const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this cluster.
        std::size_t memoryUsage() const;

    private:
        CCluster(std::size_t index,
                 const CNormalMeanPrecConjugate& prior,
                 const CNaturalBreaksClassifier& structure);

    private:
        //! A unique identifier for this cluster.
        std::size_t m_Index;

        //! The data representing this cluster.
        CNormalMeanPrecConjugate m_Prior;

        //! The data representing the internal structure of this cluster.
        CNaturalBreaksClassifier m_Structure;
    };

    using TClusterVec = std::vector<CCluster>;
    using TClusterVecItr = TClusterVec::iterator;
    using TClusterVecCItr = TClusterVec::const_iterator;

public:
    //! The central confidence interval on which to Winsorise.
    static const double WINSORISATION_CONFIDENCE_INTERVAL;

public:
    //! Construct a new clusterer.
    //!
    //! \param[in] dataType The type of data which will be clustered.
    //! \param[in] availableDistributions The distributions available to
    //! model the modes.
    //! \param[in] weightCalc The style of the cluster weight calculation
    //! (see maths_t::EClusterWeightCalc for details).
    //! \param[in] decayRate Controls the rate at which information is
    //! lost from the clusters.
    //! \param[in] minimumClusterFraction The minimum fractional count
    //! of points in a cluster.
    //! \param[in] minimumClusterCount The minimum count of points in a
    //! cluster.
    //! \param[in] minimumCategoryCount The minimum count of a category
    //! in the sketch to cluster.
    //! \param[in] winsorisationConfidenceInterval The central confidence
    //! interval on which to Winsorise.
    //! \param[in] splitFunc Optional callback for when a cluster is split.
    //! \param[in] mergeFunc Optional callback for when two clusters are
    CXMeansOnline1d(maths_t::EDataType dataType,
                    CAvailableModeDistributions availableDistributions,
                    maths_t::EClusterWeightCalc weightCalc,
                    double decayRate = 0.0,
                    double minimumClusterFraction = MINIMUM_CLUSTER_SPLIT_FRACTION,
                    double minimumClusterCount = MINIMUM_CLUSTER_SPLIT_COUNT,
                    double minimumCategoryCount = MINIMUM_CATEGORY_COUNT,
                    double winsorisationConfidenceInterval = WINSORISATION_CONFIDENCE_INTERVAL,
                    const TSplitFunc& splitFunc = CDoNothing(),
                    const TMergeFunc& mergeFunc = CDoNothing());

    //! Construct by traversing a state document.
    CXMeansOnline1d(const SDistributionRestoreParams& params,
                    core::CStateRestoreTraverser& traverser);

    //! Construct by traversing a state document.
    CXMeansOnline1d(const SDistributionRestoreParams& params,
                    const TSplitFunc& splitFunc,
                    const TMergeFunc& mergeFunc,
                    core::CStateRestoreTraverser& traverser);

    //! The x-means clusterer has value semantics.
    //@{
    CXMeansOnline1d(const CXMeansOnline1d& other);
    CXMeansOnline1d& operator=(const CXMeansOnline1d& other);
    //@}

    //! Efficiently swap the contents of two x-means objects.
    void swap(CXMeansOnline1d& other);

    //! \name Clusterer Contract
    //@{
    //! Get the tag name for this clusterer.
    virtual std::string persistenceTag() const;

    //! Persist state by passing information to the supplied inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Creates a copy of the clusterer.
    //!
    //! \warning Caller owns returned object.
    virtual CXMeansOnline1d* clone() const;

    //! Clear the current clusterer state.
    virtual void clear();

    //! Get the number of clusters.
    virtual std::size_t numberClusters() const;

    //! Set the type of data being clustered.
    virtual void dataType(maths_t::EDataType dataType);

    //! Set the rate at which information is aged out.
    virtual void decayRate(double decayRate);

    //! Check if the cluster identified by \p index exists.
    virtual bool hasCluster(std::size_t index) const;

    //! Get the centre of the cluster identified by \p index.
    virtual bool clusterCentre(std::size_t index, double& result) const;

    //! Get the spread of the cluster identified by \p index.
    virtual bool clusterSpread(std::size_t index, double& result) const;

    //! Gets the index of the cluster(s) to which \p point belongs
    //! together with their weighting factor.
    virtual void cluster(const double& point, TSizeDoublePr2Vec& result, double count = 1.0) const;

    //! Update the clustering with \p point and return its cluster(s)
    //! together with their weighting factor.
    virtual void add(const double& point, TSizeDoublePr2Vec& clusters, double count = 1.0);

    //! Update the clustering with \p points.
    virtual void add(const TDoubleDoublePrVec& points);

    //! Propagate the clustering forwards by \p time.
    //!
    //! The cluster priors relax back to non-informative and the
    //! cluster probabilities become less at a rate controlled by
    //! the decay rate parameter (optionally supplied to the constructor).
    //!
    //! \param[in] time The time increment to apply.
    //! \note \p time must be non negative.
    virtual void propagateForwardsByTime(double time);

    //! Sample the cluster with index \p index.
    //!
    //! \param[in] index The index of the cluster to sample.
    //! \param[in] numberSamples The desired number of samples.
    //! \param[out] samples Filled in with the samples.
    //! \return True if the cluster could be sampled and false otherwise.
    virtual bool sample(std::size_t index, std::size_t numberSamples, TDoubleVec& samples) const;

    //! Get the probability of the cluster with index \p index.
    //!
    //! \param[in] index The index of the cluster of interest.
    //! \return The probability of the cluster identified by \p index.
    virtual double probability(std::size_t index) const;

    //! Debug the memory used by the object.
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const;

    //! Get the static size of this object - used for virtual hierarchies
    virtual std::size_t staticSize() const;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const;
    //@}

    //! The total count of points.
    double count() const;

    //! Get the clusters.
    const TClusterVec& clusters() const;

    //! Print a representation of the clusters that can be plotted in octave.
    std::string printClusters() const;

    //! Get the index generator.
    CIndexGenerator& indexGenerator();

private:
    using TMinAccumulator = CBasicStatistics::COrderStatisticsStack<double, 1>;
    using TMaxAccumulator =
        CBasicStatistics::COrderStatisticsStack<double, 1, std::greater<double>>;

private:
    //! The minimum Kullback-Leibler divergence at which we'll
    //! split a cluster.
    static const double MINIMUM_SPLIT_DISTANCE;

    //! The maximum Kullback-Leibler divergence for which we'll
    //! merge two cluster. This is intended to introduce hysteresis
    //! in the cluster creation and deletion process and so should
    //! be less than the minimum split distance.
    static const double MAXIMUM_MERGE_DISTANCE;

    //! The default fraction of the minimum cluster split count
    //! for which we'll delete a cluster. This is intended to
    //! introduce hysteresis in the cluster creation and deletion
    //! process and so should be in the range (0, 1).
    static const double CLUSTER_DELETE_FRACTION;

    //! The size of the data we use to maintain cluster detail.
    static const std::size_t STRUCTURE_SIZE;

private:
    //! Restore by traversing a state document.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

    //! Get the cluster with the index \p index.
    const CCluster* cluster(std::size_t index) const;

    //! Compute the minimum split count.
    double minimumSplitCount() const;

    //! Split \p cluster if we find a good split.
    bool maybeSplit(TClusterVecItr cluster);

    //! Merge \p cluster and \p adjacentCluster if they are close enough.
    bool maybeMerge(TClusterVecItr cluster, TClusterVecItr adjacentCluster);

    //! Remove any clusters which are effectively dead.
    bool prune();

    //! Get the Winsorisation interval.
    TDoubleDoublePr winsorisationInterval() const;

private:
    //! The type of data being clustered.
    maths_t::EDataType m_DataType;

    //! The distributions available to model the clusters.
    CAvailableModeDistributions m_AvailableDistributions;

    //! The initial rate at which information is lost.
    double m_InitialDecayRate;

    //! The rate at which information is lost.
    double m_DecayRate;

    //! A measure of the length of history of the data clustered.
    double m_HistoryLength;

    //! The style of the cluster weight calculation (see maths_t::EClusterWeightCalc).
    maths_t::EClusterWeightCalc m_WeightCalc;

    //! The minimum cluster fractional count.
    double m_MinimumClusterFraction;

    //! The minimum cluster count.
    double m_MinimumClusterCount;

    //! The minimum count for a category in the sketch to cluster.
    double m_MinimumCategoryCount;

    //! The data central confidence interval on which to Winsorise.
    double m_WinsorisationConfidenceInterval;

    //! A generator of unique cluster indices.
    CIndexGenerator m_ClusterIndexGenerator;

    //! The smallest sample added to date.
    TMinAccumulator m_Smallest;

    //! The largest sample added to date.
    TMaxAccumulator m_Largest;

    //! The clusters.
    TClusterVec m_Clusters;

    friend ::CXMeansOnline1dTest;
};
}
}

#endif // INCLUDED_ml_maths_CXMeansOnline1d_h
