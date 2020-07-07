/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CMockDataProcessor.h"

#include <core/CLogger.h>

#include <api/COutputHandler.h>

CMockDataProcessor::CMockDataProcessor(ml::api::COutputHandler& outputHandler)
    : m_OutputHandler(outputHandler), m_NumRecordsHandled(0), m_WriteFieldNames(true) {
}

bool CMockDataProcessor::handleRecord(const TStrStrUMap& dataRowFields, TOptionalTime time) {
    // First time through we output the field names
    if (m_WriteFieldNames) {
        TStrVec fieldNames;
        fieldNames.reserve(dataRowFields.size());
        for (const auto& entry : dataRowFields) {
            fieldNames.push_back(entry.first);
        }

        if (m_OutputHandler.fieldNames(fieldNames) == false) {
            LOG_ERROR(<< "Unable to set field names for output:\n"
                      << this->debugPrintRecord(dataRowFields));
            return false;
        }
        m_WriteFieldNames = false;
    }

    if (m_OutputHandler.writeRow(dataRowFields, m_FieldOverrides, std::move(time)) == false) {
        LOG_ERROR(<< "Unable to write output");
        return false;
    }

    ++m_NumRecordsHandled;

    return true;
}

void CMockDataProcessor::finalise() {
}

bool CMockDataProcessor::isPersistenceNeeded(const std::string& description) const {
    if (m_NumRecordsHandled == 0) {
        LOG_DEBUG(<< "Zero records were handled - will not attempt to persist "
                  << description << ".");
        return false;
    }
    return true;
}

bool CMockDataProcessor::restoreState(ml::core::CDataSearcher& restoreSearcher,
                                      ml::core_t::TTime& completeToTime) {
    // Pass on the request in case we're chained
    if (m_OutputHandler.restoreState(restoreSearcher, completeToTime) == false) {
        return false;
    }

    return true;
}

bool CMockDataProcessor::persistStateInForeground(ml::core::CDataAdder& persister,
                                                  const std::string& descriptionPrefix) {
    // Pass on the request in case we're chained
    if (m_OutputHandler.persistStateInForeground(persister, descriptionPrefix) == false) {
        return false;
    }

    return true;
}

std::uint64_t CMockDataProcessor::numRecordsHandled() const {
    return m_NumRecordsHandled;
}
