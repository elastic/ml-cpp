/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_ml_model_CDetectionRule_h
#define INCLUDED_ml_model_CDetectionRule_h

#include <model/CRuleCondition.h>
#include <model/ImportExport.h>
#include <model/ModelTypes.h>

#include <string>
#include <vector>

namespace ml
{
namespace model
{
class CAnomalyDetectorModel;


//! \brief A rule that dictates an action to be taken when certain conditions occur.
//!
//! DESCRIPTION:\n
//! A rule describes an action to be taken and the conditions under which
//! the action should be taken. A rule has an action and one or more conditions.
//! The conditions are combined according to the rule's connective which can
//! be either OR or AND. A rule can optionally have a target field specified.
//! When such target is not specified, the rule applies to the series that is
//! checked against the rule. When a target is specified, the rule applies to
//! all series that are contained within the target. For example, if the target
//! is the partition field and no targetFieldValue is specified, then if the
//! conditions trigger the rule, the rule will apply to all series within the
//! partition. However, when no target is specified, the rule will trigger only
//! for series that are described in the conditions themselves.
class MODEL_EXPORT CDetectionRule
{
    public:
        using TRuleConditionVec = std::vector<CRuleCondition>;
        using TDouble1Vec = core::CSmallVector<double, 1>;

        //! Rule actions can apply to filtering results, skipping sampling or both.
        //! This is meant to work as a bit mask so added values should be powers of 2.
        enum ERuleAction
        {
            E_FilterResults                 = 1,
            E_SkipSampling                  = 2
        };

        enum EConditionsConnective
        {
            E_Or,
            E_And
        };
    public:

        //! Default constructor.
        //! The rule's action defaults to FILTER_RESULTS and the connective to OR.
        CDetectionRule();

        //! Set the rule's action.
        void action(int ruleAction);

        //! Set the conditions' connective.
        void conditionsConnective(EConditionsConnective connective);

        //! Add a condition.
        void addCondition(const CRuleCondition &condition);

        //! Set the target field name.
        void targetFieldName(const std::string &targetFieldName);

        //! Set the target field value.
        void targetFieldValue(const std::string &targetFieldValue);

        //! Check whether the rule applies on a series.
        //! \p action is bitwise and'ed with the m_Action member
        bool apply(ERuleAction action,
                   const CAnomalyDetectorModel &model,
                   model_t::EFeature feature,
                   const model_t::CResultType &resultType,
                   std::size_t pid,
                   std::size_t cid,
                   core_t::TTime time) const;

        //! Pretty-print the rule.
        std::string print() const;

    private:
        //! Check whether the given series is in the scope
        //! of the rule's target.
        bool isInScope(const CAnomalyDetectorModel &model,
                       std::size_t pid,
                       std::size_t cid) const;

        std::string printAction() const;
        std::string printConditionsConnective() const;

    private:
        //! The rule action. It works as a bit mask so its value
        //! may not match any of the declared enum values but the
        //! corresponding bit will be 1 when an action is enabled.
        int m_Action;

        //! The conditions that trigger the rule.
        TRuleConditionVec m_Conditions;

        //! The way the rule's conditions are logically connected (i.e. OR, AND).
        EConditionsConnective m_ConditionsConnective;

        //! The optional target field name. Empty when not specified.
        std::string m_TargetFieldName;

        //! The optional target field value. Empty when not specified.
        std::string m_TargetFieldValue;
};
}
}

#endif // INCLUDED_ml_model_CDetectionRule_h
