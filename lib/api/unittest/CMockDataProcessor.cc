/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
 */
#include "CMockDataProcessor.h"

#include <core/CLogger.h>

#include <api/COutputHandler.h>


CMockDataProcessor::CMockDataProcessor(ml::api::COutputHandler &outputHandler)
    : m_OutputHandler(outputHandler),
      m_NumRecordsHandled(0),
      m_WriteFieldNames(true) {
}

void CMockDataProcessor::newOutputStream(void) {
    m_OutputHandler.newOutputStream();
}

bool CMockDataProcessor::handleRecord(const TStrStrUMap &dataRowFields) {
    // First time through we output the field names
    if (m_WriteFieldNames) {
        TStrVec fieldNames;
        fieldNames.reserve(dataRowFields.size());
        for (const auto &entry : dataRowFields) {
            fieldNames.push_back(entry.first);
        }

        if (m_OutputHandler.fieldNames(fieldNames) == false) {
            LOG_ERROR("Unable to set field names for output:\n" <<
                      this->debugPrintRecord(dataRowFields));
            return false;
        }
        m_WriteFieldNames = false;
    }

    if (m_OutputHandler.writeRow(dataRowFields, m_FieldOverrides) == false) {
        LOG_ERROR("Unable to write output");
        return false;
    }

    ++m_NumRecordsHandled;

    return true;
}

void CMockDataProcessor::finalise(void) {
}

bool CMockDataProcessor::restoreState(ml::core::CDataSearcher &restoreSearcher,
                                      ml::core_t::TTime &completeToTime) {
    // Pass on the request in case we're chained
    if (m_OutputHandler.restoreState(restoreSearcher,
                                     completeToTime) == false) {
        return false;
    }

    return true;
}

bool CMockDataProcessor::persistState(ml::core::CDataAdder &persister) {
    // Pass on the request in case we're chained
    if (m_OutputHandler.persistState(persister) == false) {
        return false;
    }

    return true;
}

uint64_t CMockDataProcessor::numRecordsHandled(void) const {
    return m_NumRecordsHandled;
}

ml::api::COutputHandler &CMockDataProcessor::outputHandler(void) {
    return m_OutputHandler;
}

