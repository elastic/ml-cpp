/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CExpandingWindow_h
#define INCLUDED_ml_maths_CExpandingWindow_h

#include <core/CFloatStorage.h>
#include <core/CVectorRange.h>
#include <core/CoreTypes.h>

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
//! As the window expands it merges adjacent values and maintaining means of
//! merged values. It cycles through a sequence of increasing bucket lengths
//! supplied to the constructor. At the point it overflows, i.e. time since
//! the beginning of the window exceeds "size" x "maximum bucket length", it
//! re-initializes the bucketing and updates the start time.
//!
//! IMPLEMENTATION:\n
//! It is expected that the full window of values only needs to be accessed
//! infrequently. For example, this class is currently used by the test for
//! seasonal components and as such the full window of values is only accessed
//! when doing a test at the point the bucket length increases.
//!
//! Since the bucket values can constitute a significant amount of memory, they
//! are stored in deflated format. Empirically, this saves between 60% and 95%
//! of the memory of this class depending primarily on the number of populated
//! buckets.
//!
//! The CPU cost of in/deflating to update the current bucket is amortised by
//! maintaining a small buffer which is updated with new data points and only
//! flushed when full.
class MATHS_EXPORT CExpandingWindow {
public:
    using TDoubleVec = std::vector<double>;
    using TTimeVec = std::vector<core_t::TTime>;
    using TTimeCRng = core::CVectorRange<const TTimeVec>;
    using TFloatMeanAccumulator = CBasicStatistics::SSampleMean<CFloatStorage>::TAccumulator;
    using TFloatMeanAccumulatorVec = std::vector<TFloatMeanAccumulator>;
    using TPredictor = std::function<double(core_t::TTime)>;

public:
    CExpandingWindow(core_t::TTime bucketLength,
                     TTimeCRng bucketLengths,
                     std::size_t size,
                     double decayRate = 0.0,
                     bool deflate = true);

    //! Initialize by reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist state by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Get the start of the first bucket.
    core_t::TTime bucketStartTime() const;

    //! Get the current bucket length.
    core_t::TTime bucketLength() const;

    //! Get the number of bucket values.
    std::size_t size() const;

    //! Get the mean time offset of the data points added with respect to the start
    //! of the *data* bucket.
    core_t::TTime dataPointsTimeOffsetInBucket() const;

    //! Get the time of the first window value.
    core_t::TTime beginValuesTime() const;

    //! Get the end time of the windows' values.
    core_t::TTime endValuesTime() const;

    //! Get the bucket values.
    TFloatMeanAccumulatorVec values() const;

    //! Get the bucket values minus the predictions of \p predictor.
    TFloatMeanAccumulatorVec valuesMinusPrediction(const TPredictor& predictor) const;

    //! Get the bucket values in \p bucketValues minus the predictions of \p predictor.
    TFloatMeanAccumulatorVec valuesMinusPrediction(TFloatMeanAccumulatorVec bucketValues,
                                                   const TPredictor& predictor) const;

    //! Get an estimate of the within bucket value variance.
    double withinBucketVariance() const;

    //! Set the start time to \p time.
    void initialize(core_t::TTime time);

    //! Shift the start time by \p dt.
    void shiftTime(core_t::TTime dt);

    //! Age the bucket values to account for \p time elapsed time.
    void propagateForwardsByTime(double time);

    //! Add \p value at \p time.
    void add(core_t::TTime time, double value, double weight = 1.0);

    //! Check if we need to compress by increasing the bucket span.
    bool needToCompress(core_t::TTime time) const;

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed = 0) const;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const;

private:
    using TByte = unsigned char;
    using TByteVec = std::vector<TByte>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<double>::TAccumulator;
    using TSizeFloatMeanAccumulatorPr = std::pair<std::size_t, TFloatMeanAccumulator>;
    using TSizeFloatMeanAccumulatorPrVec = std::vector<TSizeFloatMeanAccumulatorPr>;

    //! \brief Inflates the bucket values for the lifetime of the object.
    class MATHS_EXPORT CScopeInflate : private core::CNonCopyable {
    public:
        CScopeInflate(const CExpandingWindow& window, bool commit);
        ~CScopeInflate();

    private:
        //! The window to inflate.
        const CExpandingWindow& m_Window;
        //! True if any buffered changes are to be committed.
        bool m_Commit;
    };

private:
    //! Get the end time of the window.
    core_t::TTime endTime() const;

    //! Convert to a compressed representation.
    void deflate(bool commit) const;

    //! Implements deflate.
    void doDeflate(bool commit);

    //! Extract from the compressed representation.
    void inflate(bool commit) const;

    //! Implements inflate.
    void doInflate(bool commit);

private:
    //! True if the bucket values are stored in deflated format.
    bool m_Deflate;

    //! The rate at which the bucket values are aged.
    double m_DecayRate;

    //! The number of buckets.
    std::size_t m_Size;

    //! The data bucket length.
    core_t::TTime m_DataBucketLength;

    //! The set of possible expanded window bucket lengths.
    TTimeCRng m_BucketLengths;

    //! The index in m_BucketLengths of the current window bucketing interval.
    std::size_t m_BucketLengthIndex = 0;

    //! The index of the current bucket.
    std::size_t m_BucketIndex = 0;

    //! The time of the first data point.
    core_t::TTime m_StartTime;

    //! A buffer used to amortize the cost of compression.
    TSizeFloatMeanAccumulatorPrVec m_BufferedValues;

    //! The total time to propagate the values forward on decompression.
    double m_BufferedTimeToPropagate = 0.0;

    //! The bucket values.
    TFloatMeanAccumulatorVec m_BucketValues;

    //! The deflated bucket values.
    TByteVec m_DeflatedBucketValues;

    //! The current bucket values variance accumulator.
    TMeanVarAccumulator m_WithinBucketVariance;

    //! The mean accumulator of the within bucket values variance.
    TMeanAccumulator m_AverageWithinBucketVariance;

    //! The mean offset of the window bucket values' in the bucket time interval.
    TFloatMeanAccumulator m_MeanOffset;
};
}
}

#endif // INCLUDED_ml_maths_CExpandingWindow_h
