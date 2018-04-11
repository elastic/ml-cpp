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

CDetectionRule::CDetectionRule()
    : m_Action(E_FilterResults), m_Conditions(), m_ConditionsConnective(E_Or), m_TargetFieldName(), m_TargetFieldValue() {
    m_Conditions.reserve(1);
}

void CDetectionRule::action(int action) {
    m_Action = action;
}

void CDetectionRule::conditionsConnective(EConditionsConnective connective) {
    m_ConditionsConnective = connective;
}

void CDetectionRule::addCondition(const CRuleCondition& condition) {
    m_Conditions.push_back(condition);
}

void CDetectionRule::targetFieldName(const std::string& targetFieldName) {
    m_TargetFieldName = targetFieldName;
}

void CDetectionRule::targetFieldValue(const std::string& targetFieldValue) {
    m_TargetFieldValue = targetFieldValue;
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

    if (this->isInScope(model, pid, cid) == false) {
        return false;
    }

    for (std::size_t i = 0; i < m_Conditions.size(); ++i) {
        bool conditionResult = m_Conditions[i].test(model, feature, resultType, !m_TargetFieldName.empty(), pid, cid, time);
        switch (m_ConditionsConnective) {
        case E_Or:
            if (conditionResult == true) {
                return true;
            }
            break;
        case E_And:
            if (conditionResult == false) {
                return false;
            }
            break;
        }
    }

    switch (m_ConditionsConnective) {
    case E_Or:
        return false;
    case E_And:
        return true;
    }
    return false;
}

bool CDetectionRule::isInScope(const CAnomalyDetectorModel& model, std::size_t pid, std::size_t cid) const {
    if (m_TargetFieldName.empty() || m_TargetFieldValue.empty()) {
        return true;
    }

    const CDataGatherer& gatherer = model.dataGatherer();
    if (m_TargetFieldName == gatherer.partitionFieldName()) {
        return m_TargetFieldValue == gatherer.partitionFieldValue();
    } else if (m_TargetFieldName == gatherer.personFieldName()) {
        return m_TargetFieldValue == gatherer.personName(pid);
    } else if (m_TargetFieldName == gatherer.attributeFieldName()) {
        return m_TargetFieldValue == gatherer.attributeName(cid);
    } else {
        LOG_ERROR(<< "Unexpected targetFieldName = " << m_TargetFieldName);
    }
    return false;
}

std::string CDetectionRule::print() const {
    std::string result = this->printAction();
    if (m_TargetFieldName.empty() == false) {
        result += " (" + m_TargetFieldName;
        if (m_TargetFieldValue.empty() == false) {
            result += ":" + m_TargetFieldValue;
        }
        result += ")";
    }
    result += " IF ";
    for (std::size_t i = 0; i < m_Conditions.size(); ++i) {
        result += m_Conditions[i].print();
        if (i < m_Conditions.size() - 1) {
            result += " ";
            result += this->printConditionsConnective();
            result += " ";
        }
    }
    return result;
}

std::string CDetectionRule::printAction() const {
    std::string result;
    if (E_FilterResults & m_Action) {
        result += "FILTER_RESULTS";
    }
    if (E_SkipSampling & m_Action) {
        if (result.empty() == false) {
            result += " AND ";
        }
        result += "SKIP_SAMPLING";
    }
    return result;
}

std::string CDetectionRule::printConditionsConnective() const {
    switch (m_ConditionsConnective) {
    case E_And:
        return "AND";
    case E_Or:
        return "OR";
    }
    return std::string();
}
}
}
