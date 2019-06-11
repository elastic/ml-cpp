/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CClusterer_h
#define INCLUDED_ml_maths_CClusterer_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CTypeTraits.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include <stdint.h>

namespace ml {
namespace maths {

//! \brief Factors out the non-template part of CClusterer for improved
//! compile times.
class MATHS_EXPORT CClustererTypes {
public:
    //! A no-op to provide a default for the split and merge callbacks.
    class CDoNothing {
    public:
        void operator()(std::size_t, std::size_t, std::size_t) const {}
    };

    // Callback function signature for when clusters are split.
    using TSplitFunc = std::function<void(std::size_t, std::size_t, std::size_t)>;

    // Callback function signature for when clusters are merged.
    using TMergeFunc = std::function<void(std::size_t, std::size_t, std::size_t)>;

    //! Generates unique cluster indices.
    class MATHS_EXPORT CIndexGenerator {
    public:
        //! Create a new  generator.
        CIndexGenerator();

        //! Restore by traversing a state document
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

        //! Persist state by passing information to the supplied inserter
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Deep copy this index generator.
        CIndexGenerator deepCopy() const;

        //! Get the next available unique cluster index.
        std::size_t next() const;

        //! Recycle the specified cluster index.
        void recycle(std::size_t index);

        //! Get the memory used by this component
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this component
        std::size_t memoryUsage() const;

        //! Print the state of the heap for debug.
        std::string print() const;

    private:
        using TSizeVec = std::vector<std::size_t>;
        using TSizeVecPtr = std::shared_ptr<TSizeVec>;

    private:
        //! A heap of the next available unique indices.
        TSizeVecPtr m_IndexHeap;
    };

public:
    //! \name XML Tag Names
    //!
    //! These tag the type of clusterer for polymorphic model persistence.
    //@{
    static const std::string READABLE_X_MEANS_ONLINE_1D_TAG;
    static const std::string READABLE_K_MEANS_ONLINE_1D_TAG;
    static const std::string READABLE_X_MEANS_ONLINE_TAG;

    static const std::string X_MEANS_ONLINE_1D_TAG;
    static const std::string K_MEANS_ONLINE_1D_TAG;
    static const std::string X_MEANS_ONLINE_TAG;
    //@}
};

//! \brief Interface for clustering functionality.
//!
//! DESCRIPTION:\n
//! Abstract interface for implementing incremental n-D clustering
//! functionality. Broadly speaking this comprises functionality to:
//!   -# Add new data to the clustering.
//!   -# Generate typical samples for a cluster.
//!   -# Get the probability of a cluster.
//!
//! As such even clustering algorithms which don't implicitly compute
//! a data distribution must maintain sufficient additional state to
//! approximate the data distribution within a cluster.
//!
//! Each cluster is uniquely identified by an index.
//!
//! In addition, the client code can specify optional callback to
//! invoke when clusters are split or merged.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This defines sufficient interface for implementing a multimodal
//! priors which separate the task of modeling the modes of a distribution
//! from clustering the data (see CMultimodalPrior for more details).
//!
//! This defines an object to generate unique cluster indices, which
//! supports recycling indices to avoid overflowing std::size_t, since
//! this is requirement for all implementations.
template<typename POINT>
class CClusterer : public CClustererTypes {
public:
    using TPointVec = std::vector<POINT>;
    using TPointPrecise = typename SPromoted<POINT>::Type;
    using TPointPreciseVec = std::vector<TPointPrecise>;
    using TPointPreciseDoublePr = std::pair<TPointPrecise, double>;
    using TPointPreciseDoublePrVec = std::vector<TPointPreciseDoublePr>;
    using TSizeDoublePr = std::pair<std::size_t, double>;
    using TSizeDoublePr2Vec = core::CSmallVector<TSizeDoublePr, 2>;

public:
    //! Create a new clusterer.
    //!
    //! \param splitFunc Optional callback for when a cluster is split.
    //! \param mergeFunc Optional callback for when two clusters are merged.
    explicit CClusterer(const TSplitFunc& splitFunc = CDoNothing(),
                        const TMergeFunc& mergeFunc = CDoNothing())
        : m_SplitFunc(splitFunc), m_MergeFunc(mergeFunc) {}

    virtual ~CClusterer() {}

    //! \name Clusterer Contract
    //@{
    //! Get the tag name for this clusterer.
    virtual std::string persistenceTag(bool readableTags) const = 0;

    //! Creates a copy of the clusterer.
    //!
    //! Persist state by passing information to the supplied inserter
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

    //! \warning Caller owns returned object.
    virtual CClusterer* clone() const = 0;

    //! Clear the current clusterer state.
    virtual void clear() = 0;

    //! Get the number of clusters.
    virtual std::size_t numberClusters() const = 0;

    //! Set the type of data being clustered.
    virtual void dataType(maths_t::EDataType dataType) = 0;

    //! Set the rate at which information is aged out.
    virtual void decayRate(double decayRate) = 0;

    //! Check if the cluster identified by \p index exists.
    virtual bool hasCluster(std::size_t index) const = 0;

    //! Get the centre of the cluster identified by \p index.
    virtual bool clusterCentre(std::size_t index, TPointPrecise& result) const = 0;

    //! Get the spread of the cluster identified by \p index.
    virtual bool clusterSpread(std::size_t index, double& result) const = 0;

    //! Gets the index of the cluster(s) to which \p point belongs
    //! together with their weighting factors.
    virtual void cluster(const TPointPrecise& point,
                         TSizeDoublePr2Vec& result,
                         double count = 1.0) const = 0;

    //! Add a point without caring about its cluster.
    void add(const TPointPrecise& point, double count = 1.0) {
        TSizeDoublePr2Vec clusters;
        this->add(point, clusters, count);
    }

    //! Update the clustering with \p point and return its cluster(s)
    //! together with their weighting factors.
    virtual void
    add(const TPointPrecise& point, TSizeDoublePr2Vec& clusters, double count = 1.0) = 0;

    //! Update the clustering with \p points.
    void add(const TPointPreciseVec& points) {
        TPointPreciseDoublePrVec weightedPoints;
        weightedPoints.reserve(points.size());
        for (std::size_t i = 0u; i < points.size(); ++i) {
            weightedPoints.push_back(TPointPreciseDoublePr(points[i], 1.0));
        }
        this->add(weightedPoints);
    }

    //! Update the clustering with \p points.
    virtual void add(const TPointPreciseDoublePrVec& points) = 0;

    //! Propagate the clustering forwards by \p time.
    //!
    //! The cluster priors relax back to non-informative and the
    //! cluster probabilities become less at a rate controlled by
    //! the decay rate parameter (optionally supplied to the constructor).
    //!
    //! \param time The time increment to apply.
    virtual void propagateForwardsByTime(double time) = 0;

    //! Sample the cluster with index \p index.
    //!
    //! \param index The index of the cluster to sample.
    //! \param numberSamples The desired number of samples.
    //! \param samples Filled in with the samples.
    //! \return True if the cluster could be sampled and false otherwise.
    virtual bool sample(std::size_t index,
                        std::size_t numberSamples,
                        TPointPreciseVec& samples) const = 0;

    //! Get the probability of the cluster with the index \p index.
    //!
    //! \param index The index of the cluster of interest.
    //! \return The probability of the cluster identified by \p index.
    virtual double probability(std::size_t index) const = 0;

    //! Get a checksum for this object.
    virtual uint64_t checksum(uint64_t seed = 0) const = 0;

    //! Get the memory used by this component
    virtual void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const = 0;

    //! Get the memory used by this component
    virtual std::size_t memoryUsage() const = 0;

    //! Get the static size of this object - used for virtual hierarchies
    virtual std::size_t staticSize() const = 0;
    //@}

    //! Get the callback function to invoke when a cluster is split.
    const TSplitFunc& splitFunc() const { return m_SplitFunc; }

    //! Set the callback function to invoke when a cluster is split.
    void splitFunc(const TSplitFunc& value) { m_SplitFunc = value; }

    //! Get the callback function to invoke when two clusters are merged.
    const TMergeFunc& mergeFunc() const { return m_MergeFunc; }

    //! Set the callback function to invoke when two clusters are merged.
    void mergeFunc(const TSplitFunc& value) { m_MergeFunc = value; }

protected:
    //! Swap the CClusterer state of two derived objects.
    void swap(CClusterer& other) {
        boost::swap(m_SplitFunc, other.m_SplitFunc);
        boost::swap(m_MergeFunc, other.m_MergeFunc);
    }

private:
    //! An optional callback function to invoke when a cluster is split.
    TSplitFunc m_SplitFunc;

    //! An optional callback function to invoke when two clusters are merged.
    TMergeFunc m_MergeFunc;
};

using CClusterer1d = CClusterer<double>;
}
}

#endif // INCLUDED_ml_maths_CClusterer_h
