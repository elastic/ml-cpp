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

#ifndef INCLUDED_ml_model_CDetectionRule_h
#define INCLUDED_ml_model_CDetectionRule_h

#include <core/CoreTypes.h>

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
    using TCallback = std::function<void(CAnomalyDetectorModel&, core_t::TTime)>;

    //! Rule actions can apply to skip results, skip model updates, evaluate callback
    //! function or several of the actions.
    //! This is meant to work as a bit mask so added values should be powers of 2.
    enum ERuleAction {
        E_SkipResult = 1,
        E_SkipModelUpdate = 2,
        E_TimeShift = 4
    };

public:
    //! Set the rule's action.
    void action(int ruleAction);

    //! Get the rule's action.
    int action() const;

    //! Adds a requirement for \p field not to be in \p filter for the rule to apply
    void includeScope(const std::string& field, const core::CPatternSet& filter);

    //! Adds a requirement for \p field not to be in \p filter for the rule to apply
    void excludeScope(const std::string& field, const core::CPatternSet& filter);

    //! Add a condition.
    void addCondition(const CRuleCondition& condition);

    //! Set callback function to apply some action to a supplied time series model.
    void setCallback(TCallback cb);

    //! Check whether the rule applies on a series.
    //! \p action is bitwise and'ed with the m_Action member
    bool apply(ERuleAction action,
               const CAnomalyDetectorModel& model,
               model_t::EFeature feature,
               const model_t::CResultType& resultType,
               std::size_t pid,
               std::size_t cid,
               core_t::TTime time) const;

    //! Executes the callback function for anomaly detection on the \p model if all
    //! conditions are satisfied.
    void executeCallback(CAnomalyDetectorModel& model, core_t::TTime time) const;

    //! Add a time shift callback function with the specified \p timeShift amount.
    void addTimeShift(core_t::TTime timeShift);

    //! Pretty-print the rule.
    std::string print() const;

    //! Checksum the rule.
    std::uint64_t checksum() const;

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

    //! Callback function to apply a change to a model based on the rule action.
    TCallback m_Callback;

    //! The time shift to apply to the model.
    core_t::TTime m_TimeShift{0};
};
}
}

#endif // INCLUDED_ml_model_CDetectionRule_h
