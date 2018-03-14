/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2018 Elasticsearch BV. All Rights Reserved.
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

#ifndef INCLUDED_ml_maths_CExpandingWindow_h
#define INCLUDED_ml_maths_CExpandingWindow_h

#include <core/CFloatStorage.h>
#include <core/CoreTypes.h>
#include <core/CVectorRange.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <functional>
#include <vector>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}

namespace maths {

//! \brief Implements a fixed memory expanding time window.
//!
//! DESCRIPTION:\n
//! As the window expands it compresses by merging adjacent values
//! and maintaining means of merged values. It cycles through a
//! sequence of increasing compression factors, which are determined
//! by a sequence of increasing bucketing lengths supplied to the
//! constructor. At the point it overflows, i.e. time since the
//! beginning of the window exceeds "size" x "maximum bucket length",
//! it will re-initialize the bucketing and update the start time.
class MATHS_EXPORT CExpandingWindow {
    public:
        using TDoubleVec = std::vector<double>;
        using TTimeVec = std::vector<core_t::TTime>;
        using TTimeCRng = core::CVectorRange<const TTimeVec>;
        using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
        using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
        using TPredictor = std::function<double (core_t::TTime)>;

    public:
        CExpandingWindow(core_t::TTime bucketLength,
                         TTimeCRng bucketLengths,
                         std::size_t size,
                         double decayRate = 0.0);

        //! Initialize by reading state from \p traverser.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to \p inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Get the start time of the sketch.
        core_t::TTime startTime() const;

        //! Get the end time of the sketch.
        core_t::TTime endTime() const;

        //! Get the current bucket length.
        core_t::TTime bucketLength() const;

        //! Get the bucket values.
        const TFloatMeanAccumulatorVec &values() const;

        //! Get the bucket values minus the values from \p trend.
        TFloatMeanAccumulatorVec valuesMinusPrediction(const TPredictor &predictor) const;

        //! Set the start time to \p time.
        void initialize(core_t::TTime time);

        //! Age the bucket values to account for \p time elapsed time.
        void propagateForwardsByTime(double time);

        //! Add \p value at \p time.
        void add(core_t::TTime time, double value, double weight = 1.0);

        //! Check if we need to compress by increasing the bucket span.
        bool needToCompress(core_t::TTime time) const;

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed = 0) const;

        //! Debug the memory used by this object.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this object.
        std::size_t memoryUsage() const;

    private:
        //! The rate at which the bucket values are aged.
        double                   m_DecayRate;

        //! The data bucketing length.
        core_t::TTime            m_BucketLength;

        //! The bucket lengths to test.
        TTimeCRng                m_BucketLengths;

        //! The index in m_BucketLengths of the current bucketing interval.
        std::size_t              m_BucketLengthIndex;

        //! The time of the first data point.
        core_t::TTime            m_StartTime;

        //! The bucket values.
        TFloatMeanAccumulatorVec m_BucketValues;

        //! The mean value time modulo the data bucketing length.
        TFloatMeanAccumulator    m_MeanOffset;
};

}
}

#endif // INCLUDED_ml_maths_CExpandingWindow_h
