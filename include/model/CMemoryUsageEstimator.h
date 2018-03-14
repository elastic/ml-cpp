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

#ifndef INCLUDED_ml_core_CMemoryUsageEstimator_h
#define INCLUDED_ml_core_CMemoryUsageEstimator_h

#include <core/CMemory.h>

#include <model/ImportExport.h>

#include <boost/array.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/optional.hpp>

#include <utility>

namespace ml {
namespace core {
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace model {
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
class MODEL_EXPORT CMemoryUsageEstimator {
    public:
        //! Enumeration of the components included in the memory estimate.
        enum EComponent {
            E_People = 0,
            E_Attributes,
            E_Correlations,
            E_NumberPredictors
        };
        typedef boost::array<std::size_t, E_NumberPredictors> TSizeArray;
        typedef boost::optional<std::size_t>                  TOptionalSize;

    public:
        //! Constructor
        CMemoryUsageEstimator(void);

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
        std::size_t memoryUsage(void) const;

        //! Persist this component.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

        //! Restore this component.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

    private:
        typedef std::pair<TSizeArray, std::size_t>       TSizeArraySizePr;
        typedef boost::circular_buffer<TSizeArraySizePr> TSizeArraySizePrBuf;
        typedef TSizeArraySizePrBuf::const_iterator      TSizeArraySizePrBufCItr;

    private:
        //! Get the maximum amount by which we'll extrapolate the memory usage.
        std::size_t maximumExtrapolation(EComponent component) const;

    private:
        //! The map of memory component values -> memory usage values
        TSizeArraySizePrBuf m_Values;

        //! The number of times estimate has been called since the last
        //! real value was added
        std::size_t         m_NumEstimatesSinceValue;
};

} // model
} // ml

#endif //INCLUDED_ml_core_CMemoryUsageEstimator_h
