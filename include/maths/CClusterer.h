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

#ifndef INCLUDED_ml_maths_CClusterer_h
#define INCLUDED_ml_maths_CClusterer_h

#include <core/CContainerPrinter.h>
#include <core/CLogger.h>
#include <core/CMemory.h>
#include <core/CPersistUtils.h>
#include <core/CSmallVector.h>
#include <core/CStatePersistInserter.h>
#include <core/CStateRestoreTraverser.h>

#include <maths/CTypeConversions.h>
#include <maths/ImportExport.h>
#include <maths/MathsTypes.h>

#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>
#include <functional>

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
                void operator()(std::size_t,
                                std::size_t,
                                std::size_t) const
                {}
        };

        // Callback function signature for when clusters are split.
        typedef std::function<void (std::size_t, std::size_t, std::size_t)> TSplitFunc;

        // Callback function signature for when clusters are merged.
        typedef std::function<void (std::size_t, std::size_t, std::size_t)> TMergeFunc;

        //! Generates unique cluster indices.
        class MATHS_EXPORT CIndexGenerator {
            public:
                //! Create a new  generator.
                CIndexGenerator(void);

                //! Restore by traversing a state document
                bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

                //! Persist state by passing information to the supplied inserter
                void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

                //! Deep copy this index generator.
                CIndexGenerator deepCopy(void) const;

                //! Get the next available unique cluster index.
                std::size_t next(void) const;

                //! Recycle the specified cluster index.
                void recycle(std::size_t index);

                //! Get the memory used by this component
                void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

                //! Get the memory used by this component
                std::size_t memoryUsage(void) const;

                //! Print the state of the heap for debug.
                std::string print(void) const;

            private:
                typedef std::vector<std::size_t>    TSizeVec;
                typedef boost::shared_ptr<TSizeVec> TSizeVecPtr;

            private:
                //! A heap of the next available unique indices.
                TSizeVecPtr m_IndexHeap;
        };

    public:
        //! \name XML Tag Names
        //!
        //! These tag the type of clusterer for polymorphic model persistence.
        //@{
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
        typedef boost::shared_ptr<CClusterer>        TClustererPtr;
        typedef std::vector<POINT>                   TPointVec;
        typedef typename SPromoted<POINT>::Type      TPointPrecise;
        typedef std::vector<TPointPrecise>           TPointPreciseVec;
        typedef std::pair<TPointPrecise, double>     TPointPreciseDoublePr;
        typedef std::vector<TPointPreciseDoublePr>   TPointPreciseDoublePrVec;
        typedef std::pair<std::size_t, double>       TSizeDoublePr;
        typedef core::CSmallVector<TSizeDoublePr, 2> TSizeDoublePr2Vec;

    public:
        //! Create a new clusterer.
        //!
        //! \param splitFunc Optional callback for when a cluster is split.
        //! \param mergeFunc Optional callback for when two clusters are merged.
        explicit CClusterer(const TSplitFunc &splitFunc = CDoNothing(),
                            const TMergeFunc &mergeFunc = CDoNothing()) :
            m_SplitFunc(splitFunc),
            m_MergeFunc(mergeFunc)
        {}

        virtual ~CClusterer(void) {}

        //! \name Clusterer Contract
        //@{
        //! Get the tag name for this clusterer.
        virtual std::string persistenceTag(void) const = 0;

        //! Creates a copy of the clusterer.
        //!
        //! Persist state by passing information to the supplied inserter
        virtual void acceptPersistInserter(core::CStatePersistInserter &inserter) const = 0;

        //! \warning Caller owns returned object.
        virtual CClusterer *clone(void) const = 0;

        //! Clear the current clusterer state.
        virtual void clear(void) = 0;

        //! Get the number of clusters.
        virtual std::size_t numberClusters(void) const = 0;

        //! Set the type of data being clustered.
        virtual void dataType(maths_t::EDataType dataType) = 0;

        //! Set the rate at which information is aged out.
        virtual void decayRate(double decayRate) = 0;

        //! Check if the cluster identified by \p index exists.
        virtual bool hasCluster(std::size_t index) const = 0;

        //! Get the centre of the cluster identified by \p index.
        virtual bool clusterCentre(std::size_t index,
                                   TPointPrecise &result) const = 0;

        //! Get the spread of the cluster identified by \p index.
        virtual bool clusterSpread(std::size_t index,
                                   double &result) const = 0;

        //! Gets the index of the cluster(s) to which \p point belongs
        //! together with their weighting factors.
        virtual void cluster(const TPointPrecise &point,
                             TSizeDoublePr2Vec &result,
                             double count = 1.0) const = 0;

        //! Add a point without caring about its cluster.
        void add(const TPointPrecise &point, double count = 1.0) {
            TSizeDoublePr2Vec clusters;
            this->add(point, clusters, count);
        }

        //! Update the clustering with \p point and return its cluster(s)
        //! together with their weighting factors.
        virtual void add(const TPointPrecise &point,
                         TSizeDoublePr2Vec &clusters,
                         double count = 1.0) = 0;

        //! Update the clustering with \p points.
        void add(const TPointPreciseVec &points) {
            TPointPreciseDoublePrVec weightedPoints;
            weightedPoints.reserve(points.size());
            for (std::size_t i = 0u; i < points.size(); ++i) {
                weightedPoints.push_back(TPointPreciseDoublePr(points[i], 1.0));
            }
            this->add(weightedPoints);
        }

        //! Update the clustering with \p points.
        virtual void add(const TPointPreciseDoublePrVec &points) = 0;

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
                            TPointPreciseVec &samples) const = 0;

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
        virtual std::size_t memoryUsage(void) const = 0;

        //! Get the static size of this object - used for virtual hierarchies
        virtual std::size_t staticSize(void) const = 0;
        //@}

        //! Get the callback function to invoke when a cluster is split.
        const TSplitFunc &splitFunc(void) const {
            return m_SplitFunc;
        }

        //! Set the callback function to invoke when a cluster is split.
        void splitFunc(const TSplitFunc &value) {
            m_SplitFunc = value;
        }

        //! Get the callback function to invoke when two clusters are merged.
        const TMergeFunc &mergeFunc(void) const {
            return m_MergeFunc;
        }

        //! Set the callback function to invoke when two clusters are merged.
        void mergeFunc(const TSplitFunc &value) {
            m_MergeFunc = value;
        }

    protected:
        //! Swap the CClusterer state of two derived objects.
        void swap(CClusterer &other) {
            boost::swap(m_SplitFunc, other.m_SplitFunc);
            boost::swap(m_MergeFunc, other.m_MergeFunc);
        }

    private:
        //! An optional callback function to invoke when a cluster is split.
        TSplitFunc m_SplitFunc;

        //! An optional callback function to invoke when two clusters are merged.
        TMergeFunc m_MergeFunc;
};

typedef CClusterer<double> CClusterer1d;

}
}

#endif // INCLUDED_ml_maths_CClusterer_h
