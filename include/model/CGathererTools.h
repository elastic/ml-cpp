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

#ifndef INCLUDED_ml_model_CGathererTools_h
#define INCLUDED_ml_model_CGathererTools_h

#include <core/CMemory.h>
#include <core/CSmallVector.h>
#include <core/CStoredStringPtr.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CQuantileSketch.h>

#include <model/CBucketQueue.h>
#include <model/CDataClassifier.h>
#include <model/CMetricMultivariateStatistic.h>
#include <model/CSampleGatherer.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/optional.hpp>
#include <boost/unordered_map.hpp>

#include <memory>
#include <string>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {

//! \brief A collection of utility functionality for the CDataGatherer
//! and CBucketGatherer hierarchies.
//!
//! DESCRIPTION:\n
//! A collection of utility functions primarily intended for use by the
//! the CDataGatherer and CBucketGatherer hierarchies.
//!
//! IMPLEMENTATION DECISIONS:\n
//! This class is really just a proxy for a namespace, but a class has
//! been intentionally used to provide a single point for the declaration
//! and definition of utility functions within the model library. As such
//! all member functions should be static and it should be state-less.
//! If your functionality doesn't fit this pattern just make it a nested
//! class.
class MODEL_EXPORT CGathererTools {
public:
    using TDoubleVec = std::vector<double>;
    using TOptionalDouble = boost::optional<double>;
    using TSampleVec = std::vector<CSample>;
    using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMedianAccumulator =
        maths::CFixedQuantileSketch<maths::CQuantileSketch::E_PiecewiseConstant, 30>;
    using TMinAccumulator = maths::CBasicStatistics::COrderStatisticsStack<double, 1u>;
    using TMaxAccumulator =
        maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double>>;
    using TVarianceAccumulator = maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TMultivariateMeanAccumulator = CMetricMultivariateStatistic<TMeanAccumulator>;
    using TMultivariateMinAccumulator = CMetricMultivariateStatistic<TMinAccumulator>;
    using TMultivariateMaxAccumulator = CMetricMultivariateStatistic<TMaxAccumulator>;

    //! \brief Mean arrival time gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the mean time between
    //! measurements.
    class MODEL_EXPORT CArrivalTimeGatherer {
    public:
        using TAccumulator = TMeanAccumulator;

    public:
        //! The earliest possible time.
        static const core_t::TTime FIRST_TIME;

    public:
        CArrivalTimeGatherer();

        //! Get the mean arrival time in this bucketing interval.
        TOptionalDouble featureData() const;

        //! Update the state with a new measurement.
        //!
        //! \param[in] time The time of the measurement.
        inline void add(core_t::TTime time) { this->add(time, 1); }

        //! Update the state with a measurement count.
        //!
        //! \param[in] time The end time of the \p count messages.
        //! \param[in] count The count of measurements.
        inline void add(core_t::TTime time, unsigned int count) {
            if (m_LastTime == FIRST_TIME) {
                m_LastTime = time;
            } else {
                m_Value.add(static_cast<double>(time - m_LastTime) /
                            static_cast<double>(count));
                m_LastTime = time;
            }
        }

        //! Update the state to represent the start of a new bucket.
        void startNewBucket();

        //! \name Persistence
        //@{
        //! Persist state by passing information to the supplied inserter
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Create from part of an XML document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
        //@}

        //! Get the checksum of this gatherer.
        uint64_t checksum() const;

        //! Print this gatherer for debug.
        std::string print() const;

    private:
        //! The last time a message was added.
        core_t::TTime m_LastTime;

        //! The mean time between messages received in the current
        //! bucketing interval.
        TAccumulator m_Value;
    };

    //! \brief Mean statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the arithmetic mean of
    //! a fixed number of measurements, which are supplied to the add
    //! function.
    //!
    //! This also computes the mean of all measurements in the current
    //! bucketing interval.
    using TMeanGatherer = CSampleGatherer<TMeanAccumulator, model_t::E_IndividualMeanByPerson>;

    //! \brief Multivariate mean statistic gatherer.
    //!
    //! See TMeanGatherer for details.
    using TMultivariateMeanGatherer =
        CSampleGatherer<TMultivariateMeanAccumulator, model_t::E_IndividualMeanByPerson>;

    //! \brief Median statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the median of a fixed number
    //! of measurements, which are supplied to the add function.
    using TMedianGatherer =
        CSampleGatherer<TMedianAccumulator, model_t::E_IndividualMedianByPerson>;

    // TODO Add multivariate median.

    //! \brief Minimum statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the minimum of a fixed number
    //! of measurements, which are supplied to the add function.
    //!
    //! This also computes the minimum of all measurements in the current
    //! bucketing interval.
    using TMinGatherer = CSampleGatherer<TMinAccumulator, model_t::E_IndividualMinByPerson>;

    //! \brief Multivariate minimum statistic gatherer.
    //!
    //! See TMinGatherer for details.
    using TMultivariateMinGatherer =
        CSampleGatherer<TMultivariateMinAccumulator, model_t::E_IndividualMinByPerson>;

    //! \brief Maximum statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the maximum of a fixed number
    //! of measurements, which are supplied to the add function.
    //!
    //! This also computes the maximum of all measurements in the current
    //! bucketing interval.
    using TMaxGatherer = CSampleGatherer<TMaxAccumulator, model_t::E_IndividualMaxByPerson>;

    //! \brief Multivariate maximum statistic gatherer.
    //!
    //! See TMaxGatherer for details.
    using TMultivariateMaxGatherer =
        CSampleGatherer<TMultivariateMaxAccumulator, model_t::E_IndividualMaxByPerson>;

    //! \brief Variance statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the variance of a fixed number
    //! of measurements, which are supplied to the add function.
    //!
    //! This also computes the variance of all measurements in the current
    //! bucketing interval.
    using TVarianceGatherer =
        CSampleGatherer<TVarianceAccumulator, model_t::E_IndividualVarianceByPerson>;

    // TODO Add multivariate variance.

    //! \brief Bucket sum gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the sum of a metric quantity
    //! in a bucketing interval.
    class MODEL_EXPORT CSumGatherer {
    public:
        using TDouble1Vec = core::CSmallVector<double, 1>;
        using TStrVec = std::vector<std::string>;
        using TStrVecCItr = TStrVec::const_iterator;
        using TOptionalStr = boost::optional<std::string>;
        using TOptionalStrVec = std::vector<TOptionalStr>;
        using TSampleVecQueue = CBucketQueue<TSampleVec>;
        using TSampleVecQueueItr = TSampleVecQueue::iterator;
        using TSampleVecQueueCItr = TSampleVecQueue::const_iterator;
        using TStoredStringPtrDoubleUMap =
            boost::unordered_map<core::CStoredStringPtr, double>;
        using TStoredStringPtrDoubleUMapCItr = TStoredStringPtrDoubleUMap::const_iterator;
        using TStoredStringPtrDoubleUMapQueue = CBucketQueue<TStoredStringPtrDoubleUMap>;
        using TStoredStringPtrDoubleUMapQueueCRItr =
            TStoredStringPtrDoubleUMapQueue::const_reverse_iterator;
        using TStoredStringPtrDoubleUMapQueueVec = std::vector<TStoredStringPtrDoubleUMapQueue>;
        using TStoredStringPtrVec = std::vector<core::CStoredStringPtr>;

    public:
        CSumGatherer(const SModelParams& params,
                     std::size_t dimension,
                     core_t::TTime startTime,
                     core_t::TTime bucketLength,
                     TStrVecCItr beginInfluencers,
                     TStrVecCItr endInfluencers);

        //! Get the dimension of the underlying statistic.
        std::size_t dimension() const;

        //! Get the feature data for the current bucketing interval.
        SMetricFeatureData featureData(core_t::TTime time,
                                       core_t::TTime bucketLength,
                                       const TSampleVec& emptySample) const;

        //! Returns false.
        bool sample(core_t::TTime time, unsigned int sampleCount);

        //! Update the state with a new measurement.
        //!
        //! \param[in] time The time of \p value.
        //! \param[in] value The measurement value.
        //! \param[in] influences The influencing field values which
        //! label \p value.
        void add(core_t::TTime time,
                 const TDouble1Vec& value,
                 unsigned int count,
                 unsigned int /*sampleCount*/,
                 const TStoredStringPtrVec& influences) {
            TSampleVec& sum = m_BucketSums.get(time);
            if (sum.empty()) {
                core_t::TTime bucketLength = m_BucketSums.bucketLength();
                sum.push_back(CSample(maths::CIntegerTools::floor(time, bucketLength),
                                      TDoubleVec(1, 0.0), 1.0, 0.0));
            }
            (sum[0].value())[0] += value[0];
            sum[0].count() += static_cast<double>(count);
            for (std::size_t i = 0u; i < influences.size(); ++i) {
                if (!influences[i]) {
                    continue;
                }
                TStoredStringPtrDoubleUMap& sums = m_InfluencerBucketSums[i].get(time);
                sums[influences[i]] += value[0];
            }
        }

        //! Update the state to represent the start of a new bucket.
        void startNewBucket(core_t::TTime time);

        //! Reset bucket.
        void resetBucket(core_t::TTime bucketStart);

        //! \name Persistence
        //@{
        //! Persist state by passing information to the supplied inserter
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Create from part of a state document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
        //@}

        //! Get the checksum of this gatherer.
        uint64_t checksum() const;

        //! Debug the memory used by this gatherer.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this gatherer.
        std::size_t memoryUsage() const;

        //! Print this gatherer for debug.
        std::string print() const;

        //! Is the gatherer holding redundant data?
        bool isRedundant(core_t::TTime samplingCutoffTime) const;

    private:
        //! Classifies the sum series.
        CDataClassifier m_Classifier;

        //! The sum for each bucket within the latency window.
        TSampleVecQueue m_BucketSums;

        //! The sum for each influencing field value and bucket within
        //! the latency window.
        TStoredStringPtrDoubleUMapQueueVec m_InfluencerBucketSums;
    };
};
}
}

#endif // INCLUDED_ml_model_CGathererTools_h
