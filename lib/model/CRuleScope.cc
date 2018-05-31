/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#include <model/CRuleScope.h>

#include <core/CPatternSet.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>

namespace ml {
namespace model {

CRuleScope::CRuleScope() : m_Scope() {
}

void CRuleScope::include(std::string field, const core::CPatternSet& filter) {
    m_Scope.push_back({field, TPatternSetCRef(filter), E_Include});
}

void CRuleScope::exclude(std::string field, const core::CPatternSet& filter) {
    m_Scope.push_back({field, TPatternSetCRef(filter), E_Exclude});
}

bool CRuleScope::check(const CAnomalyDetectorModel& model, std::size_t pid, std::size_t cid) const {

    const CDataGatherer& gatherer = model.dataGatherer();
    for (auto& scopeField : m_Scope) {
        bool containsValue{false};
        if (scopeField.first == gatherer.partitionFieldName()) {
            containsValue = scopeField.second.get().contains(gatherer.partitionFieldValue());
        } else if (scopeField.first == gatherer.personFieldName()) {
            containsValue = scopeField.second.get().contains(gatherer.personName(pid));
        } else if (scopeField.first == gatherer.attributeFieldName()) {
            containsValue = scopeField.second.get().contains(gatherer.attributeName(cid));
        } else {
            LOG_ERROR(<< "Unexpected scoped field = " << scopeField.first);
            return false;
        }

        if ((scopeField.third == E_Include && containsValue == false) ||
            (scopeField.third == E_Exclude && containsValue)) {
            return false;
        }
    }
    return true;
}

std::string CRuleScope::print() const {
    std::string result{""};
    if (m_Scope.empty() == false) {
        auto itr = m_Scope.begin();
        while (itr != m_Scope.end()) {
            result += "'" + itr->first + "'";
            if (itr->third == E_Include) {
                result += " IN ";
            } else {
                result += " NOT IN ";
            }
            result += "FILTER";

            ++itr;
            if (itr != m_Scope.end()) {
                result += " AND ";
            }
        }
    }
    return result;
}
}
}
