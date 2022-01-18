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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesMultibucketFeatures_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesMultibucketFeatures_h

#include <core/CMemory.h>
#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/common/CBasicStatistics.h>
#include <maths/common/CBasicStatisticsPersist.h>
#include <maths/common/CLinearAlgebraFwd.h>
#include <maths/common/MathsTypes.h>

#include <maths/time_series/CTimeSeriesMultibucketFeaturesFwd.h>
#include <maths/time_series/ImportExport.h>

#include <boost/circular_buffer.hpp>

#include <cmath>
#include <cstdint>
#include <memory>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace time_series {

//! \brief Defines features on collections of time series values.
//!
//! DESCRIPTION:\n
//! The intention of these is to provide useful features for performing anomaly
//! detection. Specifically, unusual values of certain properties of extended
//! intervals of a time series are often the most interesting events in a time
//! series from a user's perspective.
template<typename T>
class CTimeSeriesMultibucketFeature {
public:
    using Type = T;
    using TType1Vec = core::CSmallVector<Type, 1>;
    using TWeightsAry1Vec = core::CSmallVector<maths_t::TWeightsAry<Type>, 1>;
    using TType1VecTWeightAry1VecPr = std::pair<TType1Vec, TWeightsAry1Vec>;
    using TPtr = std::unique_ptr<CTimeSeriesMultibucketFeature>;

public:
    virtual ~CTimeSeriesMultibucketFeature() = default;

    //! Clone this feature.
    virtual TPtr clone() const = 0;

    //! Get the feature value.
    virtual const TType1VecTWeightAry1VecPr& value() const = 0;

    //! Get the correlation of this feature with the bucket value.
    virtual double correlationWithBucketValue() const = 0;

    //! Clear the feature state.
    virtual void clear() = 0;

    //! Update the window with \p values at \p time.
    virtual void add(core_t::TTime time,
                     core_t::TTime bucketLength,
                     const TType1Vec& values,
                     const TWeightsAry1Vec& weights) = 0;

    //! Compute a checksum for this object.
    virtual std::uint64_t checksum(std::uint64_t seed = 0) const = 0;

    //! Debug the memory used by this object.
    virtual void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const = 0;

    //! Get the static size of object.
    virtual std::size_t staticSize() const = 0;

    //! Get the memory used by this object.
    virtual std::size_t memoryUsage() const = 0;

    //! Initialize reading state from \p traverser.
    virtual bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) = 0;

    //! Persist by passing information to \p inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;
};

template<typename T, typename STORAGE>
class CTimeSeriesMultibucketMeanImpl;

//! \brief A scalar valued multi-bucket mean feature.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesMultibucketScalarMean final
    : public CTimeSeriesMultibucketScalarFeature {
public:
    using Type = CTimeSeriesMultibucketScalarFeature::Type;

public:
    explicit CTimeSeriesMultibucketScalarMean(std::size_t length = 0);
    CTimeSeriesMultibucketScalarMean(const CTimeSeriesMultibucketScalarMean& other);
    ~CTimeSeriesMultibucketScalarMean() override;

    CTimeSeriesMultibucketScalarMean(CTimeSeriesMultibucketScalarMean&&) noexcept;
    CTimeSeriesMultibucketScalarMean& operator=(CTimeSeriesMultibucketScalarMean&&) noexcept;
    CTimeSeriesMultibucketScalarMean& operator=(const CTimeSeriesMultibucketScalarMean&);

    //! Clone this feature.
    TPtr clone() const override;

    //! Get the feature value.
    const TType1VecTWeightAry1VecPr& value() const override;

    //! Get the correlation of this feature with the bucket value.
    double correlationWithBucketValue() const override;

    //! Clear the feature state.
    void clear() override;

    //! Update the window with \p values at \p time.
    void add(core_t::TTime time,
             core_t::TTime bucketLength,
             const TType1Vec& values,
             const TWeightsAry1Vec& weights) override;

    //! Compute a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the static size of object.
    std::size_t staticSize() const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

private:
    using TImpl = CTimeSeriesMultibucketMeanImpl<Type, common::CFloatStorage>;
    using TImplPtr = std::unique_ptr<TImpl>;

private:
    TImplPtr m_Impl;
};

//! \brief A vector valued multi-bucket mean feature.
class MATHS_TIME_SERIES_EXPORT CTimeSeriesMultibucketVectorMean final
    : public CTimeSeriesMultibucketVectorFeature {
public:
    using Type = CTimeSeriesMultibucketVectorFeature::Type;

public:
    explicit CTimeSeriesMultibucketVectorMean(std::size_t length = 0);
    CTimeSeriesMultibucketVectorMean(const CTimeSeriesMultibucketVectorMean& other);
    ~CTimeSeriesMultibucketVectorMean() override;

    CTimeSeriesMultibucketVectorMean(CTimeSeriesMultibucketVectorMean&&) noexcept;
    CTimeSeriesMultibucketVectorMean& operator=(CTimeSeriesMultibucketVectorMean&&) noexcept;
    CTimeSeriesMultibucketVectorMean& operator=(const CTimeSeriesMultibucketVectorMean&);

    //! Clone this feature.
    TPtr clone() const override;

    //! Get the raw feature value.
    const TType1VecTWeightAry1VecPr& value() const override;

    //! Get the correlation of this feature with the bucket value.
    double correlationWithBucketValue() const override;

    //! Clear the feature state.
    void clear() override;

    //! Update the window with \p values at \p time.
    void add(core_t::TTime time,
             core_t::TTime bucketLength,
             const TType1Vec& values,
             const TWeightsAry1Vec& weights) override;

    //! Compute a checksum for this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const override;

    //! Debug the memory used by this object.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const override;

    //! Get the static size of object.
    std::size_t staticSize() const override;

    //! Get the memory used by this object.
    std::size_t memoryUsage() const override;

    //! Initialize reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser) override;

    //! Persist by passing information to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const override;

private:
    using TImpl = CTimeSeriesMultibucketMeanImpl<Type, common::CVector<common::CFloatStorage>>;
    using TImplPtr = std::unique_ptr<TImpl>;

private:
    TImplPtr m_Impl;
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesMultibucketFeatures_h
