/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CDetectionRule_h
#define INCLUDED_ml_model_CDetectionRule_h

#include <model/CRuleCondition.h>
#include <model/CRuleScope.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <string>
#include <vector>

namespace ml {
namespace model {
class CAnomalyDetectorModel;

//! \brief A rule that dictates an action to be taken when certain conditions occur.
//!
//! DESCRIPTION:\n
//! A rule describes actions to be taken  when the scope and conditions are met.
//! A rule has one or more actions, a scope and zero or more conditions.
//! The scope dictates to which series the rule applies.
//! When conditions are present, they dictate to which results the rule applies
//! depending the result's values. Multiple conditions are combined with a logical AND.
class MODEL_EXPORT CDetectionRule {

public:
    using TRuleConditionVec = std::vector<CRuleCondition>;

    //! Rule actions can apply to skip results, skip model updates, or both.
    //! This is meant to work as a bit mask so added values should be powers of 2.
    enum ERuleAction { E_SkipResult = 1, E_SkipModelUpdate = 2 };

public:
    //! Set the rule's action.
    void action(int ruleAction);

    //! Adds a requirement for \p field not to be in \p filter for the rule to apply
    void includeScope(const std::string& field, const core::CPatternSet& filter);

    //! Adds a requirement for \p field not to be in \p filter for the rule to apply
    void excludeScope(const std::string& field, const core::CPatternSet& filter);

    //! Add a condition.
    void addCondition(const CRuleCondition& condition);

    //! Check whether the rule applies on a series.
    //! \p action is bitwise and'ed with the m_Action member
    bool apply(ERuleAction action,
               const CAnomalyDetectorModel& model,
               model_t::EFeature feature,
               const model_t::CResultType& resultType,
               std::size_t pid,
               std::size_t cid,
               core_t::TTime time) const;

    //! Pretty-print the rule.
    std::string print() const;

private:
    std::string printAction() const;

private:
    //! The rule action. It works as a bit mask so its value
    //! may not match any of the declared enum values but the
    //! corresponding bit will be 1 when an action is enabled.
    int m_Action{E_SkipResult};

    //! The rule scope.
    CRuleScope m_Scope;

    //! The conditions that trigger the rule.
    TRuleConditionVec m_Conditions;
};
}
}

#endif // INCLUDED_ml_model_CDetectionRule_h
