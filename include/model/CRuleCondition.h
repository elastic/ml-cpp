/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CRuleCondition_h
#define INCLUDED_ml_model_CRuleCondition_h

#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <boost/ref.hpp>

#include <string>

namespace ml {
namespace core {
class CPatternSet;
}
namespace model {
class CAnomalyDetectorModel;

//! \brief A numeric condition that may trigger a rule.
//!
//! DESCRIPTION:\n
//! A condition has a type that determines the calculation
//! of the value of the condition and the type of comparison
//! that will be performed. The specified fieldName/fieldValue,
//! when present, determines the series against which the
//! condition is checked.
class MODEL_EXPORT CRuleCondition {
public:
    using TPatternSetCRef = boost::reference_wrapper<const core::CPatternSet>;

public:
    enum ERuleConditionAppliesTo {
        E_Actual,
        E_Typical,
        E_DiffFromTypical,
        E_Time
    };

    enum EConditionOperator { E_LT, E_LTE, E_GT, E_GTE };

    struct SCondition {
        SCondition(EConditionOperator op, double threshold);

        bool test(double value) const;

        EConditionOperator s_Op;
        double s_Threshold;
    };

public:
    //! Default constructor.
    CRuleCondition();

    //! Set which value the condition applies to.
    void appliesTo(ERuleConditionAppliesTo appliesTo);

    //! Get the numerical condition.
    SCondition& condition();

    //! Pretty-print the condition.
    std::string print() const;

    //! Test the condition against a series.
    bool test(const CAnomalyDetectorModel& model,
              model_t::EFeature feature,
              const model_t::CResultType& resultType,
              std::size_t pid,
              std::size_t cid,
              core_t::TTime time) const;

private:
    std::string print(ERuleConditionAppliesTo appliesTo) const;
    std::string print(EConditionOperator op) const;

private:
    //! The value the condition applies to.
    ERuleConditionAppliesTo m_AppliesTo;

    //! The numerical condition.
    SCondition m_Condition;
};
}
}

#endif // INCLUDED_ml_model_CRuleCondition_h
