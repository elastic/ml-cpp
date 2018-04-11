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

#ifndef INCLUDED_ml_maths_CSeasonalComponentAdaptiveBucketing_h
#define INCLUDED_ml_maths_CSeasonalComponentAdaptiveBucketing_h

#include <core/CMemory.h>

#include <maths/CAdaptiveBucketing.h>
#include <maths/CBasicStatistics.h>
#include <maths/CRegression.h>
#include <maths/ImportExport.h>

#include <cstddef>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
class CSeasonalTime;

//! \brief An adaptive bucketing of the value of a periodic function.
//!
//! DESCRIPTION:\n
//! See CAdaptiveBucketing for details.
class MATHS_EXPORT CSeasonalComponentAdaptiveBucketing : private CAdaptiveBucketing {
public:
    using CAdaptiveBucketing::TFloatMeanAccumulatorVec;
    using TDoubleRegression = CRegression::CLeastSquaresOnline<1, double>;
    using TRegression = CRegression::CLeastSquaresOnline<1, CFloatStorage>;

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

    //! Check if the bucketing has been initialized.
    bool initialized() const;

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

    //! Get the number of buckets.
    std::size_t size() const;

    //! Clear the contents of this bucketing and recover any
    //! allocated memory.
    void clear();

    //! Shift the regressions' time origin to \p time.
    void shiftOrigin(core_t::TTime time);

    //! Shift the regressions' ordinates by \p shift.
    void shiftLevel(double shift);

    //! Shift the regressions' gradients by \p shift.
    void shiftSlope(double shift);

    //! Add the function value at \p time.
    //!
    //! \param[in] time The time of \p value.
    //! \param[in] value The value of the function at \p time.
    //! \param[in] prediction The prediction for \p value.
    //! \param[in] weight The weight of function point. The smaller
    //! this is the less influence it has on the bucket.
    void add(core_t::TTime time, double value, double prediction, double weight = 1.0);

    //! Get the time provider.
    const CSeasonalTime& time() const;

    //! Set the rate at which the bucketing loses information.
    void decayRate(double value);

    //! Get the rate at which the bucketing loses information.
    double decayRate() const;

    //! Age the bucket values to account for \p time elapsed time.
    void propagateForwardsByTime(double time, bool meanRevert = false);

    //! Get the minimum permitted bucket length.
    double minimumBucketLength() const;

    //! Refine the bucket end points to minimize the maximum averaging
    //! error in any bucket.
    //!
    //! \param[in] time The time at which to refine.
    void refine(core_t::TTime time);

    //! The count in the bucket containing \p time.
    double count(core_t::TTime time) const;

    //! Get the regression to use at \p time.
    const TRegression* regression(core_t::TTime time) const;

    //! Get a set of knot points and knot point values to use for
    //! interpolating the bucket values.
    //!
    //! \param[in] time The time at which to get the knot points.
    //! \param[in] boundary Controls the style of start and end knots.
    //! \param[out] knots Filled in with the knot points to interpolate.
    //! \param[out] values Filled in with the values at \p knots.
    //! \param[out] variances Filled in with the variances at \p knots.
    //! \return True if there are sufficient knot points to interpolate
    //! and false otherwise.
    bool knots(core_t::TTime time,
               CSplineTypes::EBoundaryCondition boundary,
               TDoubleVec& knots,
               TDoubleVec& values,
               TDoubleVec& variances) const;

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

    //! \name Test Functions
    //@{
    //! Get the bucket end points.
    const TFloatVec& endpoints() const;

    //! Get the total count of in the bucketing.
    double count() const;

    //! Get the bucket regression predictions at \p time.
    TDoubleVec values(core_t::TTime time) const;

    //! Get the bucket variances.
    TDoubleVec variances() const;
    //@}

private:
    using TSeasonalTimePtr = boost::shared_ptr<CSeasonalTime>;

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
    void refresh(const TFloatVec& endpoints);

    //! Check if \p time is in the this component's window.
    virtual bool inWindow(core_t::TTime time) const;

    //! Add the function value at \p time.
    virtual void add(std::size_t bucket, core_t::TTime time, double value, double weight);

    //! Get the offset w.r.t. the start of the bucketing of \p time.
    virtual double offset(core_t::TTime time) const;

    //! The count in \p bucket.
    virtual double count(std::size_t bucket) const;

    //! Get the predicted value for the \p bucket at \p time.
    virtual double predict(std::size_t bucket, core_t::TTime time, double offset) const;

    //! Get the variance of \p bucket.
    virtual double variance(std::size_t bucket) const;

    //! Get the interval which has been observed at \p time.
    double observedInterval(core_t::TTime time) const;

private:
    //! The time provider.
    TSeasonalTimePtr m_Time;

    //! The buckets.
    TBucketVec m_Buckets;
};

//! Create a free function which will be found by Koenig lookup.
inline void swap(CSeasonalComponentAdaptiveBucketing& lhs,
                 CSeasonalComponentAdaptiveBucketing& rhs) {
    lhs.swap(rhs);
}
}
}

#endif // INCLUDED_ml_maths_CSeasonalComponentAdaptiveBucketing_h
