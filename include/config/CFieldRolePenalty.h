/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_config_CFieldRolePenalty_h
#define INCLUDED_ml_config_CFieldRolePenalty_h

#include <core/CoreTypes.h>

#include <config/ConfigTypes.h>
#include <config/CPenalty.h>
#include <config/ImportExport.h>

#include <boost/shared_ptr.hpp>

#include <cstddef>
#include <vector>

namespace ml
{
namespace config
{
class CCategoricalDataSummaryStatistics;
class CNumericDataSummaryStatistics;

//! \brief Encapsulates the fact numeric fields can't be used for certain
//! roles, such as partitioning fields.
//!
//! \note This is a penalty for detectors based on just a single field
//! and its role in that detector. Since a whole subset of detectors can
//! share a single field for a given role then objects of this hierarchy
//! are penalty functions which are constant on the set of detectors for
//! which a given field and its role are fixed.
class CONFIG_EXPORT CCantBeNumeric : public CPenalty
{
    public:
        CCantBeNumeric(const CAutoconfigurerParams &params);

        //! Create a copy on the heap.
        virtual CCantBeNumeric *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

    private:
        //! Sets \p penalty to 0.0 for numerics and a no-op otherwise.
        virtual void penaltyFromMe(const CFieldStatistics &stats,
                                   double &penalty,
                                   std::string &description) const;
};

//! \brief Encapsulates the fact that categorical fields can't be used
//! for certain roles, such as the argument of the mean function.
//!
//! \note This is a penalty for detectors based on just a single field
//! and its role in that detector. Since a whole subset of detectors can
//! share a single field for a given role then objects of this hierarchy
//! are penalty functions which are constant on the set of detectors for
//! which a given field and its role are fixed.
class CONFIG_EXPORT CCantBeCategorical : public CPenalty
{
    public:
        CCantBeCategorical(const CAutoconfigurerParams &params);

        //! Create a copy on the heap.
        virtual CCantBeCategorical *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

    private:
        //! Sets \p penalty to 0.0 for categorical and a no-op otherwise.
        virtual void penaltyFromMe(const CFieldStatistics &stats,
                                   double &penalty,
                                   std::string &description) const;
};

//! \brief A penalty which stops unary categorical fields being used
//! to partition the data.
//!
//! \note This is a penalty for detectors based on just a single field
//! and its role in that detector. Since a whole subset of detectors can
//! share a single field for a given role then objects of this hierarchy
//! are penalty functions which are constant on the set of detectors for
//! which a given field and its role are fixed.
class CONFIG_EXPORT CDontUseUnaryField : public CPenalty
{
    public:
        CDontUseUnaryField(const CAutoconfigurerParams &params);

        //! Create a copy on the heap.
        virtual CDontUseUnaryField *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

    private:
        //! Sets \p penalty to 0.0 for categorical with a single category
        //! and a no-op otherwise.
        virtual void penaltyFromMe(const CFieldStatistics &stats,
                                   double &penalty,
                                   std::string &description) const;
};

//! \brief A penalty based on the a specified range of penalized distinct
//! counts for a categorical field.
//!
//! \note This is a penalty for detectors based on just a single field
//! and its role in that detector. Since a whole subset of detectors can
//! share a single field for a given role then objects of this hierarchy
//! are penalty functions which are constant on the set of detectors for
//! which a given field and its role are fixed.
class CONFIG_EXPORT CDistinctCountThresholdPenalty : public CPenalty
{
    public:
        CDistinctCountThresholdPenalty(const CAutoconfigurerParams &params,
                                       std::size_t distinctCountForPenaltyOfOne,
                                       std::size_t distinctCountForPenaltyOfZero);

        //! Create a copy on the heap.
        virtual CDistinctCountThresholdPenalty *clone(void) const;

        //! Get the name of this penalty.
        virtual std::string name(void) const;

    private:
        //! The penalty is a piecewise continuous linear function which
        //! is constant outside interval \f$[dc_0, dc_1]\f$ and linear
        //! decreasing from 1 at \f$dc_1\f$ to 0 at \f$dc_0\f$.
        virtual void penaltyFromMe(const CFieldStatistics &stats,
                                   double &penalty,
                                   std::string &description) const;

    private:
        //! The distinct count for which the penalty is one.
        double m_DistinctCountForPenaltyOfOne;
        //! The distinct count for which the penalty is zero.
        double m_DistinctCountForPenaltyOfZero;
};

}
}

#endif // INCLUDED_ml_config_CFieldRolePenalty_h
