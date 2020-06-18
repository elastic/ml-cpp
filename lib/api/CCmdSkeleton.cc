/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <api/CCmdSkeleton.h>

#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CLogger.h>

#include <api/CDataProcessor.h>
#include <api/CInputParser.h>

#include <functional>

namespace ml {
namespace api {

CCmdSkeleton::CCmdSkeleton(core::CDataSearcher* restoreSearcher,
                           core::CDataAdder* persister,
                           CInputParser& inputParser,
                           CDataProcessor& processor)
    : m_RestoreSearcher(restoreSearcher), m_Persister(persister),
      m_InputParser(inputParser), m_Processor(processor) {
}

bool CCmdSkeleton::ioLoop() {
    if (m_RestoreSearcher == nullptr) {
        LOG_DEBUG(<< "No restoration source specified - will not attempt to restore state");
    } else {
        core_t::TTime completeToTime(0);
        if (m_Processor.restoreState(*m_RestoreSearcher, completeToTime) == false) {
            LOG_FATAL(<< "Failed to restore state");
            return false;
        }
    }

    if (m_InputParser.readStreamIntoMaps([this](const CDataProcessor::TStrStrUMap& dataRowFields) {
            return m_Processor.handleRecord(dataRowFields, CDataProcessor::TOptionalTime{});
        }) == false) {
        LOG_FATAL(<< "Failed to handle all input data");
        return false;
    }

    LOG_INFO(<< "Handled " << m_Processor.numRecordsHandled() << " records");

    // Finalise the processor so it gets a chance to write any remaining results
    m_Processor.finalise();

    return this->persistStateInForeground();
}

bool CCmdSkeleton::persistStateInForeground() {
    if (m_Persister == nullptr) {
        LOG_DEBUG(<< "No persistence sink specified - will not attempt to persist state");
        return true;
    }

    if (m_Processor.isPersistenceNeeded("state") == false) {
        return true;
    }

    // Attempt to persist state
    if (m_Processor.persistStateInForeground(
            *m_Persister, "State persisted due to job close at ") == false) {
        LOG_FATAL(<< "Failed to persist state");
        return false;
    }

    return true;
}
}
}
