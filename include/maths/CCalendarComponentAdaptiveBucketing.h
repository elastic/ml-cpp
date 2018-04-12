/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CCalendarComponentAdaptiveBucketing_h
#define INCLUDED_ml_maths_CCalendarComponentAdaptiveBucketing_h

#include <core/CMemory.h>

#include <maths/CAdaptiveBucketing.h>
#include <maths/CBasicStatistics.h>
#include <maths/CCalendarFeature.h>
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

//! \brief An adaptive bucketing of the value of a calendar component.
//!
//! DESCRIPTION:\n
//! See CAdaptiveBucketing for details.
class MATHS_EXPORT CCalendarComponentAdaptiveBucketing : private CAdaptiveBucketing {
public:
    using TFloatMeanVarAccumulator = CBasicStatistics::SSampleMeanVar<CFloatStorage>::TAccumulator;

public:
    CCalendarComponentAdaptiveBucketing();
    explicit CCalendarComponentAdaptiveBucketing(CCalendarFeature feature,
                                                 double decayRate = 0.0,
                                                 double minimumBucketLength = 0.0);
    //! Construct by traversing a state document.
    CCalendarComponentAdaptiveBucketing(double decayRate,
                                        double minimumBucketLength,
                                        core::CStateRestoreTraverser& traverser);

    //! Persist by passing information to the supplied inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Efficiently swap the contents of two bucketing objects.
    void swap(CCalendarComponentAdaptiveBucketing& other);

    //! Check if the bucketing has been initialized.
    bool initialized() const;

    //! Create a new uniform bucketing with \p n buckets.
    //!
    //! \param[in] n The number of buckets.
    bool initialize(std::size_t n);

    //! Get the number of buckets.
    std::size_t size() const;

    //! Clear the contents of this bucketing and recover any
    //! allocated memory.
    void clear();

    //! Add the function value at \p time.
    //!
    //! \param[in] time The time of \p value.
    //! \param[in] value The value of the function at \p time.
    //! \param[in] weight The weight of function point. The smaller
    //! this is the less influence it has on the bucket.
    void add(core_t::TTime time, double value, double weight = 1.0);

    //! Get the calendar feature.
    CCalendarFeature feature() const;

    //! Set the rate at which the bucketing loses information.
    void decayRate(double value);

    //! Get the rate at which the bucketing loses information.
    double decayRate() const;

    //! Age the bucket values to account for \p time elapsed time.
    void propagateForwardsByTime(double time);

    //! Get the minimum permitted bucket length.
    double minimumBucketLength() const;

    //! Refine the bucket end points to minimize the maximum averaging
    //! error in any bucket.
    //!
    //! \param[in] time The time at which to refine.
    void refine(core_t::TTime time);

    //! The count in the bucket containing \p time.
    double count(core_t::TTime time) const;

    //! Get the count of buckets with no values.
    std::size_t emptyBucketCount() const;

    //! Get the value at \p time.
    const TFloatMeanVarAccumulator* value(core_t::TTime time) const;

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

    //! Get the bucket regressions.
    TDoubleVec values(core_t::TTime time) const;

    //! Get the bucket variances.
    TDoubleVec variances() const;
    //@}

private:
    using TFloatMeanVarVec = std::vector<TFloatMeanVarAccumulator>;

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

    //! Add the function value to \p bucket.
    virtual void add(std::size_t bucket, core_t::TTime time, double value, double weight);

    //! Get the offset w.r.t. the start of the bucketing of \p time.
    virtual double offset(core_t::TTime time) const;

    //! The count in \p bucket.
    virtual double count(std::size_t bucket) const;

    //! Get the predicted value for the \p bucket at \p time.
    virtual double predict(std::size_t bucket, core_t::TTime time, double offset) const;

    //! Get the variance of \p bucket.
    virtual double variance(std::size_t bucket) const;

private:
    //! The time provider.
    CCalendarFeature m_Feature;

    //! The bucket values.
    TFloatMeanVarVec m_Values;
};

//! Create a free function which will be found by Koenig lookup.
inline void swap(CCalendarComponentAdaptiveBucketing& lhs,
                 CCalendarComponentAdaptiveBucketing& rhs) {
    lhs.swap(rhs);
}
}
}

#endif // INCLUDED_ml_maths_CCalendarComponentAdaptiveBucketing_h
