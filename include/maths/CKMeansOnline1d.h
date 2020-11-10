/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CKMeansOnline1d_h
#define INCLUDED_ml_maths_CKMeansOnline1d_h

#include <maths/CClusterer.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/ImportExport.h>

#include <vector>

namespace ml {
namespace maths {

//! \brief A single pass online clusterer which assigns points
//! to a fixed clustering of the data.
//!
//! DESCRIPTION:\n
//! This implements the CClusterer contract for the case that
//! the clusters are decided and one wants to assign new points
//! to them in an online fashion.
//!
//! Note that this is a soft clustering so that we assign the soft
//! membership of a point to a cluster based on the probability that
//! it is generated by the corresponding normal.
class MATHS_EXPORT CKMeansOnline1d final : public CClusterer1d {
public:
    using TDoubleVec = TPointPreciseVec;
    using TDoubleDoublePrVec = TPointPreciseDoublePrVec;
    using TNormalVec = std::vector<CNormalMeanPrecConjugate>;
    using TNormalVecItr = TNormalVec::iterator;
    using TNormalVecCItr = TNormalVec::const_iterator;

public:
    //! Construct a new clusterer.
    //!
    //! \param[in] clusters The seed clusters.
    CKMeansOnline1d(TNormalVec& clusters);

    //! Construct by traversing a state document.
    CKMeansOnline1d(const SDistributionRestoreParams& params,
                    core::CStateRestoreTraverser& traverser);

    //! \name Clusterer Contract
    //@{
    //! Get the tag name for this clusterer.
    const core::TPersistenceTag& persistenceTag() const override;

    //! Persist state by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

    //! Creates a copy of the clusterer.
    //!
    //! \warning Caller owns returned object.
    CKMeansOnline1d* clone() const override;

    //! Clear the current clusterer state.
    void clear() override;

    //! No-op.
    bool remove(std::size_t index) override;

    //! Get the number of clusters.
    std::size_t numberClusters() const override;

    //! Set the type of data being clustered.
    void dataType(maths_t::EDataType dataType) override;

    //! Set the rate at which information is aged out.
    void decayRate(double decayRate) override;

    //! Check if the cluster identified by \p index exists.
    bool hasCluster(std::size_t index) const override;

    //! Get the centre of the cluster identified by \p index.
    bool clusterCentre(std::size_t index, double& result) const override;

    //! Get the spread of the cluster identified by \p index.
    bool clusterSpread(std::size_t index, double& result) const override;

    //! Gets the index of the cluster(s) to which \p point belongs
    //! together with their weighting factors.
    void cluster(const double& point, TSizeDoublePr2Vec& result, double count = 1.0) const override;

    //! Update the clustering with \p point and return its cluster(s)
    //! together with their weighting factors.
    void add(const double& point, TSizeDoublePr2Vec& clusters, double count = 1.0) override;

    //! Update the clustering with \p points.
    void add(const TDoubleDoublePrVec& points) override;

    //! Propagate the clustering forwards by \p time.
    //!
    //! The cluster priors relax back to non-informative and the
    //! cluster probabilities become less at a rate controlled by
    //! the decay rate parameter (optionally supplied to the constructor).
    //!
    //! \param[in] time The time increment to apply.
    void propagateForwardsByTime(double time) override;

    //! Sample the cluster with index \p index.
    //!
    //! \param[in] index The index of the cluster to sample.
    //! \param[in] numberSamples The desired number of samples.
    //! \param[out] samples Filled in with the samples.
    //! \return True if the cluster could be sampled and false otherwise.
    bool sample(std::size_t index, std::size_t numberSamples, TDoubleVec& samples) const override;

    //! Get the probability of the cluster with the index \p index.
    //!
    //! \param[in] index The index of the cluster of interest.
    //! \return The probability of the cluster identified by \p index.
    double probability(std::size_t index) const override;

    //! Get a checksum for this object.
    std::uint64_t checksum(uint64_t seed = 0) const override;

    //! Get the memory used by this component
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the memory used by this component
    std::size_t memoryUsage() const override;

    //! Get the static size of this object - used for virtual hierarchies
    std::size_t staticSize() const override;
    //@}

private:
    //! Restore by traversing a state document.
    bool acceptRestoreTraverser(const SDistributionRestoreParams& params,
                                core::CStateRestoreTraverser& traverser);

private:
    //! The (fixed) clusters.
    TNormalVec m_Clusters;
};
}
}

#endif // INCLUDED_ml_maths_CKMeansOnline1d_h
