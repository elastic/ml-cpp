/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CStringUtils.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CRuleCondition.h>

namespace ml {
namespace model {

namespace {
const CAnomalyDetectorModel::TSizeDoublePr1Vec EMPTY_CORRELATED;
}

using TDouble1Vec = CAnomalyDetectorModel::TDouble1Vec;

CRuleCondition::CRuleCondition()
    : m_AppliesTo(E_Actual), m_Operator(E_LT), m_Value(0.0) {
}

void CRuleCondition::appliesTo(ERuleConditionAppliesTo appliesTo) {
    m_AppliesTo = appliesTo;
}

void CRuleCondition::op(ERuleConditionOperator op) {
    m_Operator = op;
}

void CRuleCondition::value(double value) {
    m_Value = value;
}

bool CRuleCondition::test(const CAnomalyDetectorModel& model,
                          model_t::EFeature feature,
                          const model_t::CResultType& resultType,
                          std::size_t pid,
                          std::size_t cid,
                          core_t::TTime time) const {

    TDouble1Vec value;
    switch (m_AppliesTo) {
    case E_Actual: {
        value = model.currentBucketValue(feature, pid, cid, time);
        break;
    }
    case E_Typical: {
        value = model.baselineBucketMean(feature, pid, cid, resultType,
                                         EMPTY_CORRELATED, time);
        if (value.empty()) {
            // Means prior is non-informative
            return false;
        }
        break;
    }
    case E_DiffFromTypical: {
        value = model.currentBucketValue(feature, pid, cid, time);
        TDouble1Vec typical = model.baselineBucketMean(feature, pid, cid, resultType,
                                                       EMPTY_CORRELATED, time);
        if (typical.empty()) {
            // Means prior is non-informative
            return false;
        }
        if (value.size() != typical.size()) {
            LOG_ERROR(<< "Cannot apply rule condition: cannot calculate difference between "
                      << "actual and typical values due to different dimensions.");
            return false;
        }
        for (std::size_t i = 0; i < value.size(); ++i) {
            value[i] = std::fabs(value[i] - typical[i]);
        }
        break;
    }
    case E_Time: {
        value.push_back(time);
        break;
    }
    }
    if (value.empty()) {
        LOG_ERROR(<< "Value for rule comparison could not be calculated");
        return false;
    }
    if (value.size() > 1) {
        LOG_ERROR(<< "Numerical rules do not support multivariate analysis");
        return false;
    }

    return this->testValue(value[0]);
}

bool CRuleCondition::testValue(double value) const {
    switch (m_Operator) {
    case E_LT:
        return value < m_Value;
    case E_LTE:
        return value <= m_Value;
    case E_GT:
        return value > m_Value;
    case E_GTE:
        return value >= m_Value;
    }
    return false;
}

std::string CRuleCondition::print() const {
    std::string result = this->print(m_AppliesTo);
    result += " " + this->print(m_Operator) + " " +
              core::CStringUtils::typeToString(m_Value);
    return result;
}

std::string CRuleCondition::print(ERuleConditionAppliesTo appliesTo) const {
    switch (appliesTo) {
    case E_Actual:
        return "ACTUAL";
    case E_Typical:
        return "TYPICAL";
    case E_DiffFromTypical:
        return "DIFF_FROM_TYPICAL";
    case E_Time:
        return "TIME";
    }
    return std::string();
}

std::string CRuleCondition::print(ERuleConditionOperator op) const {
    switch (op) {
    case E_LT:
        return "<";
    case E_LTE:
        return "<=";
    case E_GT:
        return ">";
    case E_GTE:
        return ">=";
    }
    return std::string();
}
}
}
