/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_maths_CModelWeight_h
#define INCLUDED_ml_maths_CModelWeight_h

#include <core/CLogger.h>

#include <maths/CBasicStatistics.h>
#include <maths/ImportExport.h>

#include <boost/numeric/conversion/bounds.hpp>

#include <cstdint>

namespace ml
{
namespace core
{
class CStatePersistInserter;
class CStateRestoreTraverser;
}
namespace maths
{

//! \brief Represents the probability assigned to a model in the mixture.
//!
//! Internally the log of weights are stored to allow weights to become
//! very small without underflow. When weights are retrieved in normalized
//! form any weight which is small w.r.t. the largest weight (i.e. less
//! than double_eps * largest weight) is effectively zero and the corresponding
//! model is (temporarily) removed from the collection.
class MATHS_EXPORT CModelWeight
{
    public:
        //! See core::CMemory.
        static bool dynamicSizeAlwaysZero() { return true; }

    public:
        explicit CModelWeight(double weight);

        //! Implicit conversion to read only double weight (m_Weight).
        operator double() const;

        //! Get the log of the current weight.
        double logWeight() const;

        //! Reset the log weight.
        void logWeight(double logWeight);

        //! Add the log of a factor to the weight.
        void addLogFactor(double logFactor);

        //! Age the weight by the factor \p alpha.
        void age(double alpha);

        //! Get a checksum for this object.
        uint64_t checksum(uint64_t seed) const;

        //! Restore state from part of a state document.
        bool acceptRestoreTraverser(core::CStateRestoreTraverser &traverser);

        //! Persist state by passing information to the supplied inserter.
        void acceptPersistInserter(core::CStatePersistInserter &inserter) const;

    private:
        //! The current weight and must be in the interval [0,1].
        double m_LogWeight;

        //! The value to which the weight will revert long term and must be in
        //! the interval [0,1].
        double m_LongTermLogWeight;
};

//! \brief Re-normalizes weights (so that the sum to one) on destruction.
template<typename PRIOR>
class CScopeCanonicalizeWeights : private core::CNonCopyable
{
    public:
        using TWeightPriorPr = std::pair<CModelWeight, PRIOR>;
        using TWeightPriorPrVec = std::vector<TWeightPriorPr>;

    public:
        CScopeCanonicalizeWeights(TWeightPriorPrVec &models) : m_Models(models) {}

        ~CScopeCanonicalizeWeights()
        {
            CBasicStatistics::SMax<double>::TAccumulator logMaxWeight;
            for (const auto &model : m_Models)
            {
                logMaxWeight.add(model.first.logWeight());
            }
            for (auto &model : m_Models)
            {
                model.first.logWeight(model.first.logWeight() - logMaxWeight[0]);
            }
        }

    private:
        TWeightPriorPrVec &m_Models;
};

}
}

#endif // INCLUDED_ml_maths_CModelWeight_h
