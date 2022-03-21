/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#ifndef INCLUDED_ml_maths_time_series_CCalendarComponentAdaptiveBucketing_h
#define INCLUDED_ml_maths_time_series_CCalendarComponentAdaptiveBucketing_h

#include <core/CMemory.h>

#include <maths/common/CBasicStatistics.h>

#include <maths/time_series/CAdaptiveBucketing.h>
#include <maths/time_series/CCalendarFeature.h>
#include <maths/time_series/ImportExport.h>

#include <cstddef>
#include <vector>

#include <stdint.h>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace time_series {
class CSeasonalTime;

//! \brief An adaptive bucketing of the value of a calendar component.
//!
//! DESCRIPTION:\n
//! See CAdaptiveBucketing for details.
class MATHS_TIME_SERIES_EXPORT CCalendarComponentAdaptiveBucketing final
    : public CAdaptiveBucketing {
public:
    using TFloatMeanVarAccumulator =
        common::CBasicStatistics::SSampleMeanVar<common::CFloatStorage>::TAccumulator;
    using CAdaptiveBucketing::count;

public:
    CCalendarComponentAdaptiveBucketing();
    CCalendarComponentAdaptiveBucketing(CCalendarFeature feature,
                                        core_t::TTime timeZoneOffset,
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

    //! Create a new uniform bucketing with \p n buckets.
    //!
    //! \param[in] n The number of buckets.
    bool initialize(std::size_t n);

    //! Clear the contents of this bucketing and recover any
    //! allocated memory.
    void clear();

    //! Linearly scale the bucket values by \p scale.
    void linearScale(double scale);

    //! Add the function value at \p time.
    //!
    //! \param[in] time The time of \p value.
    //! \param[in] value The value of the function at \p time.
    //! \param[in] weight The weight of function point. The smaller
    //! this is the less influence it has on the bucket.
    void add(core_t::TTime time, double value, double weight = 1.0);

    //! Age the bucket values to account for \p time elapsed time.
    void propagateForwardsByTime(double time);

    //! Get the calendar feature.
    CCalendarFeature feature() const;

    //! The count in the bucket containing \p time.
    double count(core_t::TTime time) const;

    //! Get the count of buckets with no values.
    std::size_t emptyBucketCount() const;

    //! Get the value at \p time.
    const TFloatMeanVarAccumulator* value(core_t::TTime time) const;

    //! Get a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const;

    //! Get the memory used by this component
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this component
    std::size_t memoryUsage() const;

    //! Name of component
    std::string name() const override;

    //! Check that the state is valid.
    bool isBad() const override;

private:
    using TFloatMeanVarVec = std::vector<TFloatMeanVarAccumulator>;

private:
    //! Restore by traversing a state document
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Check the state invariants after restoration
    //! Abort on failure.
    void checkRestoredInvariants() const;

    //! Compute the values corresponding to the change in end
    //! points from \p endpoints. The values are assigned based
    //! on their intersection with each bucket in the previous
    //! bucket configuration.
    //!
    //! \param[in] endpoints The old end points.
    void refresh(const TFloatVec& endpoints) override;

    //! Check if \p time is in the this component's window.
    bool inWindow(core_t::TTime time) const override;

    //! Add the function initial value at \p time.
    void addInitialValue(std::size_t bucket, core_t::TTime time, double value, double weight) override;

    //! Get the offset w.r.t. the start of the bucketing of \p time.
    double offset(core_t::TTime time) const override;

    //! Get the count in \p bucket.
    double bucketCount(std::size_t bucket) const override;

    //! Get the predicted value for \p bucket at \p time.
    double predict(std::size_t bucket, core_t::TTime time, double offset) const override;

    //! Get the variance of \p bucket.
    double variance(std::size_t bucket) const override;

    //! Split \p bucket.
    void split(std::size_t bucket) override;

private:
    //! The time provider.
    CCalendarFeature m_Feature;

    //! The timezone offset to apply to the feature (which is in GMT).
    core_t::TTime m_TimeZoneOffset{0};

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
}

#endif // INCLUDED_ml_maths_time_series_CCalendarComponentAdaptiveBucketing_h
