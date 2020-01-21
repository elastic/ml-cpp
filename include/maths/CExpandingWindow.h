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
//! As the window expands it compresses by merging adjacent values
//! and maintaining means of merged values. It cycles through a
//! sequence of increasing compression factors, which are determined
//! by a sequence of increasing bucketing lengths supplied to the
//! constructor. At the point it overflows, i.e. time since the
//! beginning of the window exceeds "size" x "maximum bucket length",
//! it will re-initialize the bucketing and update the start time.
//!
//! IMPLEMENTATION:\n
//! It is expected that the full window of values only needs to be
//! accessed infrequently. For example, this class is currently used
//! by the test for seasonal components and as such the full window
//! of values is only accessed when doing a test at the point the
//! bucketing interval expands.
//!
//! Since the bucket values can constitute a significant amount of
//! memory, one can choose to store them in deflated format. Empirically,
//! this saves between 60% and 95% of the memory of this class depending
//! primarily on the number of populated buckets.
//!
//! The CPU cost of deflation is amortised by maintaining a small buffer
//! which is update with new values and only flushed when full.
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

    //! Get the start time of the sketch.
    core_t::TTime startTime() const;

    //! Get the end time of the sketch.
    core_t::TTime endTime() const;

    //! Get the mean offset of values in the bucket.
    core_t::TTime offset() const;

    //! Get the current bucket length.
    core_t::TTime bucketLength() const;

    //! Get the number of bucket values.
    std::size_t size() const;

    //! Get the bucket values.
    TFloatMeanAccumulatorVec values() const;

    //! Get the bucket values minus the values from \p trend.
    TFloatMeanAccumulatorVec valuesMinusPrediction(const TPredictor& predictor) const;

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

    //! The data bucketing length.
    core_t::TTime m_BucketLength;

    //! The bucket lengths to test.
    TTimeCRng m_BucketLengths;

    //! The index in m_BucketLengths of the current bucketing interval.
    std::size_t m_BucketLengthIndex;

    //! The time of the first data point.
    core_t::TTime m_StartTime;

    //! A buffer used to amortize the cost of compression.
    TSizeFloatMeanAccumulatorPrVec m_BufferedValues;

    //! Get the total time to propagate the values forward on decompression.
    double m_BufferedTimeToPropagate;

    //! The bucket values.
    TFloatMeanAccumulatorVec m_BucketValues;

    //! The deflated bucket values.
    TByteVec m_DeflatedBucketValues;

    //! The mean value time modulo the data bucketing length.
    TFloatMeanAccumulator m_MeanOffset;
};
}
}

#endif // INCLUDED_ml_maths_CExpandingWindow_h
