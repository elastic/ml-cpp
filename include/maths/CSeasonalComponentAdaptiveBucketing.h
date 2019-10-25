/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CSeasonalComponentAdaptiveBucketing_h
#define INCLUDED_ml_maths_CSeasonalComponentAdaptiveBucketing_h

#include <core/CMemory.h>

#include <maths/CAdaptiveBucketing.h>
#include <maths/CBasicStatistics.h>
#include <maths/CLeastSquaresOnlineRegression.h>
#include <maths/CSeasonalTime.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <memory>
#include <vector>

#include <stdint.h>

namespace CTimeSeriesDecompositionTest {
class CNanInjector;
}

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {

//! \brief An adaptive bucketing of the value of a periodic function.
//!
//! DESCRIPTION:\n
//! See CAdaptiveBucketing for details.
class MATHS_EXPORT CSeasonalComponentAdaptiveBucketing final : public CAdaptiveBucketing {
public:
    using CAdaptiveBucketing::TFloatMeanAccumulatorVec;
    using TDoubleRegression = CLeastSquaresOnlineRegression<1, double>;
    using TRegression = CLeastSquaresOnlineRegression<1, CFloatStorage>;
    using CAdaptiveBucketing::count;

public:
    CSeasonalComponentAdaptiveBucketing();
    explicit CSeasonalComponentAdaptiveBucketing(const CSeasonalTime& time,
                                                 double decayRate = 0.0,
                                                 double minimumBucketLength = 0.0);
    CSeasonalComponentAdaptiveBucketing(const CSeasonalComponentAdaptiveBucketing& other);
    //! Construct by traversing a state document.
    CSeasonalComponentAdaptiveBucketing(double decayRate,
                                        double minimumBucketLength,
                                        core::CStateRestoreTraverser& traverser);

    //! Copy from \p rhs.
    const CSeasonalComponentAdaptiveBucketing&
    operator=(const CSeasonalComponentAdaptiveBucketing& rhs);

    //! Persist by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Efficiently swap the contents of two bucketing objects.
    void swap(CSeasonalComponentAdaptiveBucketing& other);

    //! Create a new uniform bucketing with \p n buckets.
    //!
    //! \param[in] n The number of buckets.
    bool initialize(std::size_t n);

    //! Add the function moments \f$([a_i,b_i], S_i)\f$ where
    //! \f$S_i\f$ are the means and variances of the function
    //! in the time intervals \f$([a_i,b_i])\f$.
    //!
    //! \param[in] startTime The start of the period including \p values.
    //! \param[in] endTime The end of the period including \p values.
    //! \param[in] values Time ranges and the corresponding function
    //! value moments.
    void initialValues(core_t::TTime startTime,
                       core_t::TTime endTime,
                       const TFloatMeanAccumulatorVec& values);

    //! Clear the contents of this bucketing and recover any
    //! allocated memory.
    void clear();

    //! Shift the regressions' time origin to \p time.
    void shiftOrigin(core_t::TTime time);

    //! Shift the regressions' ordinates by \p shift.
    void shiftLevel(double shift);

    //! Shift the regressions' gradients by \p shift keeping the prediction
    //! at \p time fixed.
    void shiftSlope(core_t::TTime time, double shift);

    //! Linearly scale the regressions by \p scale.
    void linearScale(double scale);

    //! Add the function value at \p time.
    //!
    //! \param[in] time The time of \p value.
    //! \param[in] value The value of the function at \p time.
    //! \param[in] prediction The prediction for \p value.
    //! \param[in] weight The weight of function point. The smaller
    //! this is the less influence it has on the bucket.
    void add(core_t::TTime time, double value, double prediction, double weight = 1.0);

    //! Age the bucket values to account for \p time elapsed time.
    void propagateForwardsByTime(double time, bool meanRevert = false);

    //! Get the time provider.
    const CSeasonalTime& time() const;

    //! Get the count in the bucket containing \p time.
    double count(core_t::TTime time) const;

    //! Get the regression to use at \p time.
    const TRegression* regression(core_t::TTime time) const;

    //! Get the common slope of the bucket regression models.
    double slope() const;

    //! Check if this regression models have enough history to predict.
    bool slopeAccurate(core_t::TTime time) const;

    //! Get a checksum for this object.
    uint64_t checksum(uint64_t seed = 0) const;

    //! Get the memory used by this component
    void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

    //! Get the memory used by this component
    std::size_t memoryUsage() const;

    //! Name of component
    std::string name() const override;

    //! Check that the state is valid.
    bool isBad() const override;

private:
    using TSeasonalTimePtr = std::unique_ptr<CSeasonalTime>;

    //! \brief The state maintained for each bucket.
    struct SBucket {
        SBucket();
        SBucket(const TRegression& regression,
                double variance,
                core_t::TTime firstUpdate,
                core_t::TTime lastUpdate);

        bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);
        void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

        uint64_t checksum(uint64_t seed) const;

        //! Check that the state is valid.
        bool isBad() const;

        TRegression s_Regression;
        CFloatStorage s_Variance;
        core_t::TTime s_FirstUpdate;
        core_t::TTime s_LastUpdate;
    };
    using TBucketVec = std::vector<SBucket>;

private:
    //! Restore by traversing a state document
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Compute the values corresponding to the change in end
    //! points from \p endpoints. The values are assigned based
    //! on their intersection with each bucket in the previous
    //! bucket configuration.
    //!
    //! \param[in] endpoints The old end points.
    void refresh(const TFloatVec& endpoints) override;

    //! Check if \p time is in the this component's window.
    bool inWindow(core_t::TTime time) const override;

    //! Add the function value at \p time.
    void add(std::size_t bucket, core_t::TTime time, double value, double weight) override;

    //! Get the offset w.r.t. the start of the bucketing of \p time.
    double offset(core_t::TTime time) const override;

    //! The count in \p bucket.
    double bucketCount(std::size_t bucket) const override;

    //! Get the predicted value for \p bucket at \p time.
    double predict(std::size_t bucket, core_t::TTime time, double offset) const override;

    //! Get the variance of \p bucket.
    double variance(std::size_t bucket) const override;

    //! Split \p bucket.
    void split(std::size_t bucket) override;

    //! Get the interval which has been observed at \p time.
    double observedInterval(core_t::TTime time) const;

private:
    //! The time provider.
    TSeasonalTimePtr m_Time;

    //! The buckets.
    TBucketVec m_Buckets;

    //! Befriend a helper class used by the unit tests
    friend class CTimeSeriesDecompositionTest::CNanInjector;
};

//! Create a free function which will be found by Koenig lookup.
inline void swap(CSeasonalComponentAdaptiveBucketing& lhs,
                 CSeasonalComponentAdaptiveBucketing& rhs) {
    lhs.swap(rhs);
}
}
}

#endif // INCLUDED_ml_maths_CSeasonalComponentAdaptiveBucketing_h
