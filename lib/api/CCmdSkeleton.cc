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
#include <api/CCmdSkeleton.h>

#include <core/CDataAdder.h>
#include <core/CDataSearcher.h>
#include <core/CLogger.h>
#include <core/CProgramCounters.h>

#include <model/ModelTypes.h>

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
        // If we restore state then that state will supply the assignment memory
        // basis, and for jobs that pre-date this field, that will be restored
        // as "unknown".  But if we are starting from scratch then we should
        // start with the model memory limit approach, as this will save the
        // Java code having to make a complicated decision to determine this.
        core::CProgramCounters::counter(counter_t::E_TSADAssignmentMemoryBasis) =
            static_cast<std::uint64_t>(model_t::E_AssignmentBasisModelMemoryLimit);
    } else {
        core_t::TTime completeToTime(0);
        if (m_Processor.restoreState(*m_RestoreSearcher, completeToTime) == false) {
            LOG_FATAL(<< "Failed to restore state");
            return false;
        }
    }

    if (m_InputParser.readStreamIntoMaps(
            [this](const CDataProcessor::TStrStrUMap& dataRowFields) {
                return m_Processor.handleRecord(dataRowFields,
                                                CDataProcessor::TOptionalTime{});
            },
            [this](const std::string& fieldName, std::string& fieldValue) {
                m_Processor.registerMutableField(fieldName, fieldValue);
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
