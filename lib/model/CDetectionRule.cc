/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CDetectionRule.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>

namespace ml {
namespace model {

void CDetectionRule::action(int action) {
    m_Action = action;
}

void CDetectionRule::includeScope(std::string field, const core::CPatternSet& filter) {
    m_Scope.include(field, filter);
}

void CDetectionRule::excludeScope(std::string field, const core::CPatternSet& filter) {
    m_Scope.exclude(field, filter);
}

void CDetectionRule::addCondition(const CRuleCondition& condition) {
    m_Conditions.push_back(condition);
}

bool CDetectionRule::apply(ERuleAction action,
                           const CAnomalyDetectorModel& model,
                           model_t::EFeature feature,
                           const model_t::CResultType& resultType,
                           std::size_t pid,
                           std::size_t cid,
                           core_t::TTime time) const {
    if (!(m_Action & action)) {
        return false;
    }

    if (m_Scope.check(model, pid, cid) == false) {
        return false;
    }

    for (const auto& condition : m_Conditions) {
        if (condition.test(model, feature, resultType, pid, cid, time) == false) {
            return false;
        }
    }

    return true;
}

std::string CDetectionRule::print() const {
    std::string result = this->printAction();
    result += " IF ";
    std::string scopeString = m_Scope.print();
    result += scopeString;
    if (scopeString.empty() == false && m_Conditions.empty() == false) {
        result += " AND ";
    }
    for (std::size_t i = 0; i < m_Conditions.size(); ++i) {
        result += m_Conditions[i].print();
        if (i < m_Conditions.size() - 1) {
            result += " AND ";
        }
    }
    return result;
}

std::string CDetectionRule::printAction() const {
    std::string result;
    if (E_SkipResult & m_Action) {
        result += "SKIP_RESULT";
    }
    if (E_SkipModelUpdate & m_Action) {
        if (result.empty() == false) {
            result += " AND ";
        }
        result += "SKIP_MODEL_UPDATE";
    }
    return result;
}
}
}
