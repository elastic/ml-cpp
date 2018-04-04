/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_core_CMemoryUsageEstimator_h
#define INCLUDED_ml_core_CMemoryUsageEstimator_h

#include <core/CMemory.h>

#include <model/ImportExport.h>

#include <boost/array.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>

#include <utility>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model
{
//! \brief Estimate memory usage based on previous model parameters.
//!
//! DESCRIPTION:\n
//! This is used to estimate memory usage based on previous measurements,
//! on the assumption that memory usage scales linearly
//! with the number of people and attributes in the model.
//!
//! IMPLEMENTATION DECISIONS:\n
//! A circular buffer is used to store previous calculations.
//! An Eigen Jacobian Single Value Decomposition technique is used to solve the
//! linear least squares system, with the values of people and attributes
//! forming the input matrix A, which is solved for the memory usage
//! calculations in vector B.
//! See http://eigen.tuxfamily.org/dox-devel/group__LeastSquares.html
class MODEL_EXPORT CMemoryUsageEstimator
{
    public:
        //! Enumeration of the components included in the memory estimate.
        enum EComponent
        {
            E_People = 0,
            E_Attributes,
            E_Correlations,
            E_NumberPredictors
        };
        using TSizeArray = boost::array<std::size_t, E_NumberPredictors>;
        using TOptionalSize = boost::optional<std::size_t>;

    public:
        //! Constructor
        CMemoryUsageEstimator();

        //! Get an estimate of the memory usage based on the given number
        //! of different factors which contribute.
        //!
        //! This can fail, for example if too many estimations have taken
        //! place, in which case a TOptionalSize() will be returned, indicating
        //! that the caller must add a real value.
        TOptionalSize estimate(const TSizeArray &predictors);

        //! Add an actual memory calculation value, along with the values of
        //! the predictors.
        void addValue(const TSizeArray &predictors, std::size_t memory);

        //! Debug the memory used by this component.
        void debugMemoryUsage(core::CMemoryUsage::TMemoryUsagePtr mem) const;

        //! Get the memory used by this component.
        std::size_t memoryUsage() const;

        //! Persist this component.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Restore this component.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

    private:
        using TSizeArraySizePr = std::pair<TSizeArray, std::size_t>;
        using TSizeArraySizePrBuf = boost::circular_buffer<TSizeArraySizePr>;
        using TSizeArraySizePrBufCItr = TSizeArraySizePrBuf::const_iterator;

    private:
        //! Get the maximum amount by which we'll extrapolate the memory usage.
        std::size_t maximumExtrapolation(EComponent component) const;

    private:
        //! The map of memory component values -> memory usage values
        TSizeArraySizePrBuf m_Values;

        //! The number of times estimate has been called since the last
        //! real value was added
        std::size_t m_NumEstimatesSinceValue;
};

} // model
} // ml

#endif //INCLUDED_ml_core_CMemoryUsageEstimator_h
