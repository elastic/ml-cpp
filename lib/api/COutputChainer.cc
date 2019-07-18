/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/COutputChainer.h>

#include <core/CLogger.h>

#include <api/CDataProcessor.h>

#include <functional>

namespace ml {
namespace api {

COutputChainer::COutputChainer(CDataProcessor& dataProcessor)
    : m_DataProcessor(dataProcessor) {
}

void COutputChainer::newOutputStream() {
    m_DataProcessor.newOutputStream();
}

bool COutputChainer::fieldNames(const TStrVec& fieldNames, const TStrVec& extraFieldNames) {
    m_FieldNames = fieldNames;

    // Only add extra field names if they're not already present
    for (TStrVecCItr iter = extraFieldNames.begin(); iter != extraFieldNames.end(); ++iter) {
        if (std::find(m_FieldNames.begin(), m_FieldNames.end(), *iter) ==
            m_FieldNames.end()) {
            m_FieldNames.push_back(*iter);
        }
    }

    m_Hashes.clear();
    m_WorkRecordFieldRefs.clear();
    m_WorkRecordFields.clear();

    if (m_FieldNames.empty()) {
        LOG_ERROR(<< "Attempt to set empty field names");
        return false;
    }

    m_Hashes.reserve(m_FieldNames.size());
    m_WorkRecordFieldRefs.reserve(m_FieldNames.size());

    // Pre-compute the hashes for each field name (assuming the hash function is
    // the same for our empty overrides map as it is for the ones provided by
    // callers)
    for (TStrVecCItr iter = m_FieldNames.begin(); iter != m_FieldNames.end(); ++iter) {
        m_Hashes.push_back(EMPTY_FIELD_OVERRIDES.hash_function()(*iter));
        m_WorkRecordFieldRefs.push_back(std::ref(m_WorkRecordFields[*iter]));
    }

    return true;
}

bool COutputChainer::writeRow(const TStrStrUMap& dataRowFields,
                              const TStrStrUMap& overrideDataRowFields) {
    if (m_FieldNames.empty()) {
        LOG_ERROR(<< "Attempt to output data before field names");
        return false;
    }

    using TStrEqualTo = std::equal_to<std::string>;
    TStrEqualTo pred;

    TPreComputedHashVecCItr preComputedHashIter = m_Hashes.begin();
    TStrRefVecCItr fieldRefIter = m_WorkRecordFieldRefs.begin();
    for (TStrVecCItr fieldNameIter = m_FieldNames.begin();
         fieldNameIter != m_FieldNames.end() && preComputedHashIter != m_Hashes.end() &&
         fieldRefIter != m_WorkRecordFieldRefs.end();
         ++fieldNameIter, ++preComputedHashIter, ++fieldRefIter) {
        TStrStrUMapCItr fieldValueIter =
            overrideDataRowFields.find(*fieldNameIter, *preComputedHashIter, pred);
        if (fieldValueIter == overrideDataRowFields.end()) {
            fieldValueIter = dataRowFields.find(*fieldNameIter, *preComputedHashIter, pred);
            if (fieldValueIter == dataRowFields.end()) {
                LOG_ERROR(<< "Output fields do not include a value for field " << *fieldNameIter);
                return false;
            }
        }

        // Use the start/length version of assign to bypass GNU copy-on-write,
        // since we don't want the strings in m_WorkRecordFields to share
        // representations with strings in our input maps.
        fieldRefIter->get().assign(fieldValueIter->second, 0,
                                   fieldValueIter->second.length());
    }

    if (m_DataProcessor.handleRecord(m_WorkRecordFields) == false) {
        LOG_ERROR(<< "Chained data processor function returned false for record:" << core_t::LINE_ENDING
                  << CDataProcessor::debugPrintRecord(m_WorkRecordFields));
        return false;
    }

    return true;
}

void COutputChainer::finalise() {
    m_DataProcessor.finalise();
}

bool COutputChainer::restoreState(core::CDataSearcher& restoreSearcher,
                                  core_t::TTime& completeToTime) {
    return m_DataProcessor.restoreState(restoreSearcher, completeToTime);
}

bool COutputChainer::persistState(core::CDataAdder& persister,
                                  const std::string& descriptionPrefix) {
    return m_DataProcessor.persistState(persister, descriptionPrefix);
}

bool COutputChainer::periodicPersistStateInBackground() {
    return m_DataProcessor.periodicPersistStateInBackground();
}

bool COutputChainer::isPersistenceNeeded(const std::string& description) const {
    return m_DataProcessor.isPersistenceNeeded(description);
}

bool COutputChainer::consumesControlMessages() {
    return true;
}
}
}
