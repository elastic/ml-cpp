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

#include <core/CoreTypes.h>
#include <core/CSmallVector.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDetectionRule.h>

namespace ml {
namespace model {

void CDetectionRule::action(int action) {
    m_Action = action;
}

int CDetectionRule::action() const {
    return m_Action;
}

void CDetectionRule::includeScope(const std::string& field, const core::CPatternSet& filter) {
    m_Scope.include(field, filter);
}

void CDetectionRule::excludeScope(const std::string& field, const core::CPatternSet& filter) {
    m_Scope.exclude(field, filter);
}

void CDetectionRule::addCondition(const CRuleCondition& condition) {
    m_Conditions.push_back(condition);
}

void CDetectionRule::setCallback(TCallback cb) {
    m_Callback = std::move(cb);
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

void CDetectionRule::executeCallback(CAnomalyDetectorModel& model, core_t::TTime time) const {

    if (m_Callback != nullptr) {
        for (const auto& condition : m_Conditions) {
            if (condition.test(time) == false) { // Disjunction
                return;
            }
        }
        m_Callback(model, time);
    }
}

void CDetectionRule::addTimeShift(core_t::TTime timeShift) {
    using TAnomalyDetectorPtrVec = core::CSmallVector<CAnomalyDetectorModel*, 2>;
    this->setCallback([
        timeShift, timeShiftApplied = TAnomalyDetectorPtrVec()
    ](CAnomalyDetectorModel & model, core_t::TTime time) mutable {
        if (std::find(timeShiftApplied.begin(), timeShiftApplied.end(), &model) ==
            timeShiftApplied.end()) {
            // When the callback is executed, the model is already in the correct time
            // interval. Hence, we need to shift the time right away.
            model.shiftTime(time, timeShift);
            timeShiftApplied.emplace_back(&model);
        }
    });
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
    if (E_TimeShift & m_Action) {
        if (result.empty() == false) {
            result += " AND ";
        }
        result += "FORCE_TIME_SHIFT";
    }
    return result;
}
}
}
