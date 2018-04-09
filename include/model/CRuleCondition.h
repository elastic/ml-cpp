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

//! \brief A condition that may trigger a rule.
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
    enum ERuleConditionType { E_CategoricalMatch, E_CategoricalComplement, E_NumericalActual, E_NumericalTypical, E_NumericalDiffAbs, E_Time };

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

    //! Set the condition type.
    void type(ERuleConditionType ruleType);

    //! Set the field name. Empty means it is not specified.
    void fieldName(const std::string& fieldName);

    //! Set the field value. Empty means it is not specified.
    void fieldValue(const std::string& fieldValue);

    //! Get the numerical condition.
    SCondition& condition();

    //! Set the value filter (used for categorical only).
    void valueFilter(const core::CPatternSet& valueFilter);

    //! Is the condition categorical?
    //! Categorical conditions are pattern match conditions i.e.
    //! E_CategoricalMatch and E_CategoricalComplement
    bool isCategorical() const;

    //! Is the condition numerical?
    bool isNumerical() const;

    //! Pretty-print the condition.
    std::string print() const;

    //! Test the condition against a series.
    bool test(const CAnomalyDetectorModel& model,
              model_t::EFeature feature,
              const model_t::CResultType& resultType,
              bool isScoped,
              std::size_t pid,
              std::size_t cid,
              core_t::TTime time) const;

private:
    bool checkCondition(const CAnomalyDetectorModel& model,
                        model_t::EFeature feature,
                        model_t::CResultType resultType,
                        std::size_t pid,
                        std::size_t cid,
                        core_t::TTime time) const;
    std::string print(ERuleConditionType type) const;
    std::string print(EConditionOperator op) const;

private:
    //! The condition type.
    ERuleConditionType m_Type;

    //! The numerical condition.
    SCondition m_Condition;

    //! The field name. Empty when not specified.
    std::string m_FieldName;

    //! The field value. Empty when not specified.
    std::string m_FieldValue;

    TPatternSetCRef m_ValueFilter;
};
}
}

#endif // INCLUDED_ml_model_CRuleCondition_h
