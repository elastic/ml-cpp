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
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>

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
    typedef std::vector<double> TDoubleVec;
    typedef boost::optional<double> TOptionalDouble;
    typedef std::vector<CSample> TSampleVec;
    typedef maths::CBasicStatistics::SSampleMean<double>::TAccumulator TMeanAccumulator;
    typedef maths::CFixedQuantileSketch<maths::CQuantileSketch::E_PiecewiseConstant, 30> TMedianAccumulator;
    typedef maths::CBasicStatistics::COrderStatisticsStack<double, 1u> TMinAccumulator;
    typedef maths::CBasicStatistics::COrderStatisticsStack<double, 1u, std::greater<double>> TMaxAccumulator;
    typedef maths::CBasicStatistics::SSampleMeanVar<double>::TAccumulator TVarianceAccumulator;
    typedef CMetricMultivariateStatistic<TMeanAccumulator> TMultivariateMeanAccumulator;
    typedef CMetricMultivariateStatistic<TMinAccumulator> TMultivariateMinAccumulator;
    typedef CMetricMultivariateStatistic<TMaxAccumulator> TMultivariateMaxAccumulator;

    //! \brief Mean arrival time gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the mean time between
    //! measurements.
    class MODEL_EXPORT CArrivalTimeGatherer {
    public:
        typedef TMeanAccumulator TAccumulator;

    public:
        //! The earliest possible time.
        static const core_t::TTime FIRST_TIME;

    public:
        CArrivalTimeGatherer(void);

        //! Get the mean arrival time in this bucketing interval.
        TOptionalDouble featureData(void) const;

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
                m_Value.add(static_cast<double>(time - m_LastTime) / static_cast<double>(count));
                m_LastTime = time;
            }
        }

        //! Update the state to represent the start of a new bucket.
        void startNewBucket(void);

        //! \name Persistence
        //@{
        //! Persist state by passing information to the supplied inserter
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        //! Create from part of an XML document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
        //@}

        //! Get the checksum of this gatherer.
        uint64_t checksum(void) const;

        //! Print this gatherer for debug.
        std::string print(void) const;

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
    typedef CSampleGatherer<TMeanAccumulator, model_t::E_IndividualMeanByPerson> TMeanGatherer;

    //! \brief Multivariate mean statistic gatherer.
    //!
    //! See TMeanGatherer for details.
    typedef CSampleGatherer<TMultivariateMeanAccumulator, model_t::E_IndividualMeanByPerson> TMultivariateMeanGatherer;

    //! \brief Median statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the median of a fixed number
    //! of measurements, which are supplied to the add function.
    typedef CSampleGatherer<TMedianAccumulator, model_t::E_IndividualMedianByPerson> TMedianGatherer;

    // TODO Add multivariate median.

    //! \brief Minimum statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the minimum of a fixed number
    //! of measurements, which are supplied to the add function.
    //!
    //! This also computes the minimum of all measurements in the current
    //! bucketing interval.
    typedef CSampleGatherer<TMinAccumulator, model_t::E_IndividualMinByPerson> TMinGatherer;

    //! \brief Multivariate minimum statistic gatherer.
    //!
    //! See TMinGatherer for details.
    typedef CSampleGatherer<TMultivariateMinAccumulator, model_t::E_IndividualMinByPerson> TMultivariateMinGatherer;

    //! \brief Maximum statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the maximum of a fixed number
    //! of measurements, which are supplied to the add function.
    //!
    //! This also computes the maximum of all measurements in the current
    //! bucketing interval.
    typedef CSampleGatherer<TMaxAccumulator, model_t::E_IndividualMaxByPerson> TMaxGatherer;

    //! \brief Multivariate maximum statistic gatherer.
    //!
    //! See TMaxGatherer for details.
    typedef CSampleGatherer<TMultivariateMaxAccumulator, model_t::E_IndividualMaxByPerson> TMultivariateMaxGatherer;

    //! \brief Variance statistic gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the variance of a fixed number
    //! of measurements, which are supplied to the add function.
    //!
    //! This also computes the variance of all measurements in the current
    //! bucketing interval.
    typedef CSampleGatherer<TVarianceAccumulator, model_t::E_IndividualVarianceByPerson> TVarianceGatherer;

    // TODO Add multivariate variance.

    //! \brief Bucket sum gatherer.
    //!
    //! DESCRIPTION:\n
    //! Wraps up the functionality to sample the sum of a metric quantity
    //! in a bucketing interval.
    class MODEL_EXPORT CSumGatherer {
    public:
        typedef core::CSmallVector<double, 1> TDouble1Vec;
        typedef std::vector<std::string> TStrVec;
        typedef TStrVec::const_iterator TStrVecCItr;
        typedef boost::optional<std::string> TOptionalStr;
        typedef std::vector<TOptionalStr> TOptionalStrVec;
        typedef CBucketQueue<TSampleVec> TSampleVecQueue;
        typedef TSampleVecQueue::iterator TSampleVecQueueItr;
        typedef TSampleVecQueue::const_iterator TSampleVecQueueCItr;
        typedef boost::unordered_map<core::CStoredStringPtr, double> TStoredStringPtrDoubleUMap;
        typedef TStoredStringPtrDoubleUMap::const_iterator TStoredStringPtrDoubleUMapCItr;
        typedef CBucketQueue<TStoredStringPtrDoubleUMap> TStoredStringPtrDoubleUMapQueue;
        typedef TStoredStringPtrDoubleUMapQueue::const_reverse_iterator TStoredStringPtrDoubleUMapQueueCRItr;
        typedef std::vector<TStoredStringPtrDoubleUMapQueue> TStoredStringPtrDoubleUMapQueueVec;
        typedef std::vector<core::CStoredStringPtr> TStoredStringPtrVec;

    public:
        CSumGatherer(const SModelParams& params,
                     std::size_t dimension,
                     core_t::TTime startTime,
                     core_t::TTime bucketLength,
                     TStrVecCItr beginInfluencers,
                     TStrVecCItr endInfluencers);

        //! Get the dimension of the underlying statistic.
        std::size_t dimension(void) const;

        //! Get the feature data for the current bucketing interval.
        SMetricFeatureData
        featureData(core_t::TTime time, core_t::TTime bucketLength, const TSampleVec& emptySample) const;

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
                sum.push_back(CSample(maths::CIntegerTools::floor(time, bucketLength), TDoubleVec(1, 0.0), 1.0, 0.0));
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
        uint64_t checksum(void) const;

        //! Debug the memory used by this gatherer.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this gatherer.
        std::size_t memoryUsage(void) const;

        //! Print this gatherer for debug.
        std::string print(void) const;

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
