/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CInputParser.h>

#include <algorithm>

namespace ml {
namespace api {

CInputParser::CInputParser(TStrVec mutableFieldNames)
    : m_MutableFieldNames(std::move(mutableFieldNames)) {
}

void CInputParser::registerMutableFields(const TRegisterMutableFieldFunc& registerFunc,
                                         TStrStrUMap& dataRowFields) const {

    if (registerFunc) {
        for (const auto& mutableFieldName : m_MutableFieldNames) {
            registerFunc(mutableFieldName, dataRowFields[mutableFieldName]);
        }
    } else {
        for (const auto& mutableFieldName : m_MutableFieldNames) {
            dataRowFields[mutableFieldName];
        }
    }
}

void CInputParser::registerMutableFields(const TRegisterMutableFieldFunc& registerFunc,
                                         TStrVec& fieldNames,
                                         TStrVec& fieldValues) const {

    // This has to be done in two passes as adding the fields could invalidate
    // the references registered - all additions must be done before any
    // registrations
    for (const auto& mutableFieldName : m_MutableFieldNames) {
        if (std::find(fieldNames.begin(), fieldNames.end(), mutableFieldName) ==
            fieldNames.end()) {
            fieldNames.push_back(mutableFieldName);
            fieldValues.emplace_back();
        }
    }
    if (registerFunc) {
        for (const auto& mutableFieldName : m_MutableFieldNames) {
            auto iter = std::find(fieldNames.begin(), fieldNames.end(), mutableFieldName);
            registerFunc(mutableFieldName, fieldValues[iter - fieldNames.begin()]);
        }
    }
}

const CInputParser::TStrVec& CInputParser::fieldNames() const {
    return m_FieldNames;
}

CInputParser::TStrVec& CInputParser::fieldNames() {
    return m_FieldNames;
}
}
}
