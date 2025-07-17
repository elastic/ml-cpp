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

#ifndef INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionBase_h
#define INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionBase_h

#include <core/CMemoryCircuitBreaker.h>
#include <core/CMemoryUsage.h>

#include <maths/common/Constants.h>

#include <maths/time_series/CTimeSeriesDecompositionInterface.h>
#include <maths/time_series/ImportExport.h>

#include <functional>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths {
namespace common {
struct STimeSeriesDecompositionRestoreParams;
}
namespace time_series {

//! \brief Base abstract class for time series decomposition components
//!
//! DESCRIPTION:\n
//! This provides the common interface and basic functionality for all time 
//! series decomposition implementations. It serves as the foundation for 
//! specialized decomposition components.
class MATHS_TIME_SERIES_EXPORT EMPTY_BASE_OPT CTimeSeriesDecompositionBase
    : public CTimeSeriesDecompositionInterface {
public:
    using TVector2x1 = CTimeSeriesDecompositionInterface::TVector2x1;
    using TDouble3Vec = CTimeSeriesDecompositionInterface::TDouble3Vec;
    using TDoubleVec = std::vector<double>;
    using TFloatMeanAccumulatorVec = CTimeSeriesDecompositionInterface::TFloatMeanAccumulatorVec;
    using TBoolVec = std::vector<bool>;
    using TComponentChangeCallback = CTimeSeriesDecompositionInterface::TComponentChangeCallback;
    using TWriteForecastResult = CTimeSeriesDecompositionInterface::TWriteForecastResult;
    using TWeights = maths_t::CUnitWeights;
    using TFilteredPredictor = std::function<double(core_t::TTime, const TBoolVec&)>;

public:
    //! \param[in] decayRate The rate at which information is lost.
    //! \param[in] bucketLength The data bucketing length.
    explicit CTimeSeriesDecompositionBase(double decayRate = 0.0,
                                          core_t::TTime bucketLength = 0);

    //! Construct from part of a state document.
    CTimeSeriesDecompositionBase(const common::STimeSeriesDecompositionRestoreParams& params,
                                 core::CStateRestoreTraverser& traverser);

    //! Virtual destructor
    virtual ~CTimeSeriesDecompositionBase() override = default;

    //! Persist state by passing information to the supplied inserter.
    virtual void acceptPersistInserter(core::CStatePersistInserter& inserter) const = 0;

    //! Set the decay rate.
    virtual void decayRate(double decayRate) override = 0;

    //! Get the decay rate.
    virtual double decayRate() const override = 0;

    //! Check if the decomposition has any initialized components.
    virtual bool initialized() const override = 0;

    //! Get the time shift which is being applied.
    virtual core_t::TTime timeShift() const override = 0;

protected:
    //! Get the bucket length
    core_t::TTime bucketLength() const;

    //! Set the bucket length
    void bucketLength(core_t::TTime bucketLength);

private:
    //! The data bucketing length.
    core_t::TTime m_BucketLength;
};
}
}
}

#endif // INCLUDED_ml_maths_time_series_CTimeSeriesDecompositionBase_h
