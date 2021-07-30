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

#include <model/CRuleScope.h>

#include <core/CPatternSet.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>

namespace ml {
namespace model {

void CRuleScope::include(const std::string& field, const core::CPatternSet& filter) {
    m_Scope.emplace_back(field, TPatternSetCRef(filter), E_Include);
}

void CRuleScope::exclude(const std::string& field, const core::CPatternSet& filter) {
    m_Scope.emplace_back(field, TPatternSetCRef(filter), E_Exclude);
}

bool CRuleScope::check(const CAnomalyDetectorModel& model, std::size_t pid, std::size_t cid) const {

    const CDataGatherer& gatherer = model.dataGatherer();
    for (const auto& scopeField : m_Scope) {
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
    return result;
}
}
}
