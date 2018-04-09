/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <core/CLogger.h>
#include <core/CPatternSet.h>
#include <core/CStringUtils.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CRuleCondition.h>

namespace ml {
namespace model {

namespace {
const CAnomalyDetectorModel::TSizeDoublePr1Vec EMPTY_CORRELATED;
const core::CPatternSet EMPTY_FILTER;
}

using TDouble1Vec = CAnomalyDetectorModel::TDouble1Vec;

CRuleCondition::SCondition::SCondition(EConditionOperator op, double threshold) : s_Op(op), s_Threshold(threshold) {
}

bool CRuleCondition::SCondition::test(double value) const {
    switch (s_Op) {
    case E_LT:
        return value < s_Threshold;
    case E_LTE:
        return value <= s_Threshold;
    case E_GT:
        return value > s_Threshold;
    case E_GTE:
        return value >= s_Threshold;
    }
    return false;
}

CRuleCondition::CRuleCondition()
    : m_Type(E_NumericalActual), m_Condition(E_LT, 0.0), m_FieldName(), m_FieldValue(), m_ValueFilter(EMPTY_FILTER) {
}

void CRuleCondition::type(ERuleConditionType ruleType) {
    m_Type = ruleType;
}

void CRuleCondition::fieldName(const std::string& fieldName) {
    m_FieldName = fieldName;
}

void CRuleCondition::fieldValue(const std::string& fieldValue) {
    m_FieldValue = fieldValue;
}

CRuleCondition::SCondition& CRuleCondition::condition() {
    return m_Condition;
}

void CRuleCondition::valueFilter(const core::CPatternSet& valueFilter) {
    m_ValueFilter = TPatternSetCRef(valueFilter);
}

bool CRuleCondition::isCategorical() const {
    return m_Type == E_CategoricalMatch || m_Type == E_CategoricalComplement;
}

bool CRuleCondition::isNumerical() const {
    return !this->isCategorical();
}

bool CRuleCondition::test(const CAnomalyDetectorModel& model,
                          model_t::EFeature feature,
                          const model_t::CResultType& resultType,
                          bool isScoped,
                          std::size_t pid,
                          std::size_t cid,
                          core_t::TTime time) const {
    const CDataGatherer& gatherer = model.dataGatherer();

    if (this->isCategorical()) {
        bool containsValue{false};
        if (m_FieldName == gatherer.partitionFieldName()) {
            containsValue = m_ValueFilter.get().contains(gatherer.partitionFieldValue());
        } else if (m_FieldName == gatherer.personFieldName()) {
            containsValue = m_ValueFilter.get().contains(gatherer.personName(pid));
        } else if (m_FieldName == gatherer.attributeFieldName()) {
            containsValue = m_ValueFilter.get().contains(gatherer.attributeName(cid));
        } else {
            LOG_ERROR("Unexpected fieldName = " << m_FieldName);
            return false;
        }

        return (m_Type == E_CategoricalComplement) ? !containsValue : containsValue;
    } else {
        if (m_FieldValue.empty() == false) {
            if (isScoped) {
                // When scoped we are checking if the rule condition applies to the entity
                // identified by m_FieldName/m_FieldValue, and we do this for all time
                // series which have resolved to check this condition.
                // Thus we ignore the supplied pid/cid and instead look up
                // the time series identifier that matches the condition's m_FieldValue.
                bool successfullyResolvedId =
                    model.isPopulation() ? gatherer.attributeId(m_FieldValue, cid) : gatherer.personId(m_FieldValue, pid);
                if (successfullyResolvedId == false) {
                    return false;
                }
            } else {
                // For numerical rules the field name may be:
                //   - empty
                //   - the person field name if the detector has only an over field or only a by field
                //   - the attribute field name if the detector has both over and by fields
                const std::string& fieldValue = model.isPopulation() && m_FieldName == gatherer.attributeFieldName()
                                                    ? gatherer.attributeName(cid)
                                                    : gatherer.personName(pid);
                if (m_FieldValue != fieldValue) {
                    return false;
                }
            }
        }
        return this->checkCondition(model, feature, resultType, pid, cid, time);
    }
}

bool CRuleCondition::checkCondition(const CAnomalyDetectorModel& model,
                                    model_t::EFeature feature,
                                    model_t::CResultType resultType,
                                    std::size_t pid,
                                    std::size_t cid,
                                    core_t::TTime time) const {
    TDouble1Vec value;
    switch (m_Type) {
    case E_CategoricalMatch:
    case E_CategoricalComplement: {
        LOG_ERROR("Should never check numerical condition for categorical rule condition");
        return false;
    }
    case E_NumericalActual: {
        value = model.currentBucketValue(feature, pid, cid, time);
        break;
    }
    case E_NumericalTypical: {
        value = model.baselineBucketMean(feature, pid, cid, resultType, EMPTY_CORRELATED, time);
        if (value.empty()) {
            // Means prior is non-informative
            return false;
        }
        break;
    }
    case E_NumericalDiffAbs: {
        value = model.currentBucketValue(feature, pid, cid, time);
        TDouble1Vec typical = model.baselineBucketMean(feature, pid, cid, resultType, EMPTY_CORRELATED, time);
        if (typical.empty()) {
            // Means prior is non-informative
            return false;
        }
        if (value.size() != typical.size()) {
            LOG_ERROR("Cannot apply rule condition: cannot calculate difference between "
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
        LOG_ERROR("Value for rule comparison could not be calculated");
        return false;
    }
    if (value.size() > 1) {
        LOG_ERROR("Numerical rules do not support multivariate analysis");
        return false;
    }

    return m_Condition.test(value[0]);
}

std::string CRuleCondition::print() const {
    std::string result = this->print(m_Type);
    if (m_FieldName.empty() == false) {
        result += "(" + m_FieldName;
        if (m_FieldValue.empty() == false) {
            result += ":" + m_FieldValue;
        }
        result += ")";
    }
    result += " ";

    if (this->isCategorical()) {
        if (m_Type == E_CategoricalComplement) {
            result += "NOT ";
        }
        result += "IN FILTER";
    } else {
        result += this->print(m_Condition.s_Op) + " " + core::CStringUtils::typeToString(m_Condition.s_Threshold);
    }
    return result;
}

std::string CRuleCondition::print(ERuleConditionType type) const {
    switch (type) {
    case E_CategoricalMatch:
    case E_CategoricalComplement:
        return "";
    case E_NumericalActual:
        return "ACTUAL";
    case E_NumericalTypical:
        return "TYPICAL";
    case E_NumericalDiffAbs:
        return "DIFF_ABS";
    case E_Time:
        return "TIME";
    }
    return std::string();
}

std::string CRuleCondition::print(EConditionOperator op) const {
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
