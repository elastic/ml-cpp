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

#ifndef INCLUDED_ml_model_CInterimBucketCorrector_h
#define INCLUDED_ml_model_CInterimBucketCorrector_h

#include <core/CMemory.h>

#include <maths/CBasicStatistics.h>
#include <maths/CTimeSeriesDecomposition.h>

#include <model/ImportExport.h>
#include <model/ModelTypes.h>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model
{

//! \brief Calculate prediction-based corrections for interim results.
//!
//! DESCRIPTION:\n
//! This is used to calculate corrections for interim results values so that
//! they are adjusted according to an estimation of how complete the bucket is.
//! This decreases the probability of false positives for functions that are
//! sensitive to the bucket completeness (e.g. count).
//!
//! IMPLEMENTATION DECISIONS:\n
//! In order to estimate the bucket completeness the overall bucket count is
//! modelled and the completeness is calculated as the ratio of the current
//! bucket count against the expected one. This method was preferred compared
//! to a time-based calculation because it makes less assumptions about the
//! distribution of events over time. The bucket count is modelled via a time
//! series decomposition. While the decomposition is not initialiased, a mean
//! accumulator is used.
class MODEL_EXPORT CInterimBucketCorrector
{
    private:
        using TMeanAccumulator = maths::CBasicStatistics::SSampleMean<double>::TAccumulator;
        using TDouble1Vec = core::CSmallVector<double, 1>;
        using TDouble10Vec = core::CSmallVector<double, 10>;

    public:
        //! Constructs an interim bucket corrector for buckets of length \p bucketLength
        CInterimBucketCorrector(core_t::TTime bucketLength);

        //! Copy constructor
        CInterimBucketCorrector(const CInterimBucketCorrector &other);

        //! Updates the model of the bucket count with a new measurement
        void update(core_t::TTime time, std::size_t bucketCount);

        //! Calculates corrections for the \p value based on the given \p mode
        //! and the estimated bucket completeness.
        //!
        //! \param[in] time The time of interest.
        //! \param[in] currentCount The total count in the bucket of interest.
        //! \param[in] mode The mode that corresponds to the given \p value.
        //! \param[in] value The value to be corrected.
        double corrections(core_t::TTime time,
                           std::size_t currentCount,
                           double mode,
                           double value) const;

        //! Calculates corrections for the \p values based on the given \p modes
        //! and the estimated bucket completeness.
        //!
        //! \param[in] time The time of interest.
        //! \param[in] currentCount The total count in the bucket of interest.
        //! \param[in] modes The modes that map to the given \p values.
        //! \param[in] values The values to be corrected.
        TDouble10Vec corrections(core_t::TTime time,
                                 std::size_t currentCount,
                                 const TDouble10Vec &modes,
                                 const TDouble10Vec &values) const;

        //! Get the memory used by the corrector
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by the corrector
        std::size_t memoryUsage() const;

        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

    private:
        //! Returns the mid point in the bucket that contains \p time.
        core_t::TTime calcBucketMidPoint(core_t::TTime time) const;

        //! Calculates an estimate of completeness for a bucket that contains
        //! \p time and whose current count is \p currentCount. The returned
        //! value is within [0.0, 1.0].
        double estimateBucketCompleteness(core_t::TTime time, std::size_t currentCount) const;

    private:
        //! The bucket length
        core_t::TTime m_BucketLength;

        //! The decomposition of the overall bucket count
        maths::CTimeSeriesDecomposition m_CountTrend;

        //! The mean statistic for the overall bucket count
        TMeanAccumulator m_CountMean;
};

}
}

#endif // INCLUDED_ml_model_CInterimBucketCorrector_h
