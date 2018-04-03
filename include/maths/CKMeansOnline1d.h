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

#ifndef INCLUDED_ml_maths_CKMeansOnline1d_h
#define INCLUDED_ml_maths_CKMeansOnline1d_h

#include <maths/CClusterer.h>
#include <maths/CNormalMeanPrecConjugate.h>
#include <maths/ImportExport.h>

#include <vector>

namespace ml
{
namespace maths
{

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
class MATHS_EXPORT CKMeansOnline1d : public CClusterer1d
{
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
        CKMeansOnline1d(TNormalVec &clusters);

        //! Construct by traversing a state document.
        CKMeansOnline1d(const SDistributionRestoreParams &params,
                        core::CStateRestoreTraverser &traverser);

        //! \name Clusterer Contract
        //@{
        //! Get the tag name for this clusterer.
        virtual std::string persistenceTag(void) const;

        //! Persist state by passing information to the supplied inserter.
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Creates a copy of the clusterer.
        //!
        //! \warning Caller owns returned object.
        virtual CKMeansOnline1d *clone(void) const;

        //! Clear the current clusterer state.
        virtual void clear(void);

        //! Get the number of clusters.
        virtual std::size_t numberClusters(void) const;

        //! Set the type of data being clustered.
        virtual void dataType(maths_t::EDataType dataType);

        //! Set the rate at which information is aged out.
        virtual void decayRate(double decayRate);

        //! Check if the cluster identified by \p index exists.
        virtual bool hasCluster(std::size_t index) const;

        //! Get the centre of the cluster identified by \p index.
        virtual bool clusterCentre(std::size_t index,
                                   double &result) const;

        //! Get the spread of the cluster identified by \p index.
        virtual bool clusterSpread(std::size_t index,
                                   double &result) const;

        //! Gets the index of the cluster(s) to which \p point belongs
        //! together with their weighting factors.
        virtual void cluster(const double &point,
                             TSizeDoublePr2Vec &result,
                             double count = 1.0) const;

        //! Update the clustering with \p point and return its cluster(s)
        //! together with their weighting factors.
        virtual void add(const double &point,
                         TSizeDoublePr2Vec &clusters,
                         double count = 1.0);

        //! Update the clustering with \p points.
        virtual void add(const TDoubleDoublePrVec &points);

        //! Propagate the clustering forwards by \p time.
        //!
        //! The cluster priors relax back to non-informative and the
        //! cluster probabilities become less at a rate controlled by
        //! the decay rate parameter (optionally supplied to the constructor).
        //!
        //! \param[in] time The time increment to apply.
        virtual void propagateForwardsByTime(double time);

        //! Sample the cluster with index \p index.
        //!
        //! \param[in] index The index of the cluster to sample.
        //! \param[in] numberSamples The desired number of samples.
        //! \param[out] samples Filled in with the samples.
        //! \return True if the cluster could be sampled and false otherwise.
        virtual bool sample(std::size_t index,
                            std::size_t numberSamples,
                            TDoubleVec &samples) const;

        //! Get the probability of the cluster with the index \p index.
        //!
        //! \param[in] index The index of the cluster of interest.
        //! \return The probability of the cluster identified by \p index.
        virtual double probability(std::size_t index) const;

        //! Get a checksum for this object.
        virtual uint64_t checksum(uint64_t seed = 0) const;

        //! Get the memory used by this component
        virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this component
        virtual std::size_t memoryUsage(void) const;

        //! Get the static size of this object - used for virtual hierarchies
        virtual std::size_t staticSize(void) const;
        //@}

    private:
        //! Restore by traversing a state document.
        bool acceptRestoreTraverser(const SDistributionRestoreParams &params,
                                    core::CStateRestoreTraverser &traverser);

    private:
        //! The (fixed) clusters.
        TNormalVec m_Clusters;
};

}
}

#endif // INCLUDED_ml_maths_CKMeansOnline1d_h
