/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CDecayRateController_h
#define INCLUDED_ml_maths_CDecayRateController_h

#include <core/CSmallVector.h>
#include <core/CoreTypes.h>

#include <maths/CBasicStatistics.h>
#include <maths/CPRNG.h>
#include <maths/ImportExport.h>

#include <array>
#include <cstdint>

namespace ml {
namespace core {
class CStateRestoreTraverser;
class CStatePersistInserter;
}
namespace maths {

//! \brief Manages the decay rate based on the data characteristics.
//!
//! DESCRIPTION:\n
//! We can use estimates of the prediction errors to understand if our
//! models are capturing the time varying components of a time series
//! and if there has recently been a significant change from the time
//! series behavior. In particular, we look at
//!   -# The ratio of the prediction bias to the prediction error.
//!   -# The ratio of the recent absolute to the long term prediction
//!      error.
//!
//! If there is a significant bias in our predictions then our model is
//! failing to capture some time varying component of the time series
//! and the best we can do is to remember less history. If the short term
//! and prediction error is large compared to the long term prediction
//! error then the system has recently undergone some state change and
//! we should re-learn the model parameters as fast as possible.
class MATHS_EXPORT CDecayRateController {
public:
    using TDouble1Vec = core::CSmallVector<double, 1>;
    using TDouble1VecVec = std::vector<TDouble1Vec>;
    using TMeanAccumulator = CBasicStatistics::SSampleMean<double>::TAccumulator;
    using TMeanAccumulator1Vec = core::CSmallVector<TMeanAccumulator, 1>;

    //! Enumerates the type of model check we can perform.
    enum EChecks {
        E_PredictionBias = 0x1,          //!< Check for prediction bias.
        E_PredictionErrorIncrease = 0x2, //!< Check for recent increases
                                         //! in the prediction errors.
        E_PredictionErrorDecrease = 0x4  //!< Check for recent decreases
                                         //! in the prediction errors.
    };

public:
    CDecayRateController();
    CDecayRateController(int checks, std::size_t dimension);

    //! Get the checks which this controller is performing.
    int checks() const;

    //! Set the checks which this controller is performing.
    void checks(int checks);

    //! Reset the errors.
    void reset();

    //! Restore by reading state from \p traverser.
    bool acceptRestoreTraverser(core::CStateRestoreTraverser& traverser);

    //! Persist by passing state to \p inserter.
    void acceptPersistInserter(core::CStatePersistInserter& inserter) const;

    //! Get the decay rate multiplier to apply and update the relevant
    //! prediction errors.
    double multiplier(const TDouble1Vec& prediction,
                      const TDouble1VecVec& predictionErrors,
                      core_t::TTime bucketLength,
                      double learnRate,
                      double decayRate);

    //! Get the current multiplier.
    double multiplier() const;

    //! Get the dimension of the time series model this controls.
    std::size_t dimension() const;

    //! Debug the memory used by this controller.
    void debugMemoryUsage(const core::CMemoryUsage::TMemoryUsagePtr& mem) const;

    //! Get the memory used by this controller.
    std::size_t memoryUsage() const;

    //! Get a checksum of this object.
    std::uint64_t checksum(std::uint64_t seed = 0) const;

private:
    using TDouble3Ary = std::array<double, 3>;

private:
    double count() const;
    double change(const TDouble3Ary& stats, core_t::TTime bucketLength) const;
    bool notControlling() const;
    bool increaseDecayRateErrorIncreasing(const TDouble3Ary& stats) const;
    bool increaseDecayRateErrorDecreasing(const TDouble3Ary& stats) const;
    bool increaseDecayRateBiased(const TDouble3Ary& stats) const;
    bool decreaseDecayRateErrorNotIncreasing(const TDouble3Ary& stats) const;
    bool decreaseDecayRateErrorNotDecreasing(const TDouble3Ary& stats) const;
    bool decreaseDecayRateNotBiased(const TDouble3Ary& stats) const;

private:
    //! The checks we perform to detect error conditions.
    int m_Checks = 0;

    //! The current target multiplier.
    double m_Target = 1.0;

    //! The cumulative multiplier applied to the decay rate.
    TMeanAccumulator m_Multiplier;

    //! A random number generator.
    CPRNG::CXorOShiro128Plus m_Rng;

    //! The mean predicted value.
    TMeanAccumulator1Vec m_PredictionMean;

    //! The mean bias in the model predictions.
    TMeanAccumulator1Vec m_Bias;

    //! The short term absolute errors in the model predictions.
    TMeanAccumulator1Vec m_RecentAbsError;

    //! The long term absolute errors in the model predictions.
    TMeanAccumulator1Vec m_HistoricalAbsError;
};
}
}

#endif // INCLUDED_ml_maths_CDecayRateController_h
