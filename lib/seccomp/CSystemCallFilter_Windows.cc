/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "seccomp/CSystemCallFilter.h"

#include <core/CLogger.h>
#include <core/CWindowsError.h>

#include <WinBase.h>


namespace ml {
namespace seccomp {

CSystemCallFilter::CSystemCallFilter() {
    HANDLE job = CreateJobObject(nullptr, nullptr);
    if (job == nullptr) {
        LOG_ERROR(<< "Failed to create Job Object: " << ml::core::CWindowsError().errorString());
    }

    JOBOBJECT_BASIC_LIMIT_INFORMATION limits;

    // Get the current job information
    if (QueryInformationJobObject(job, JobObjectBasicLimitInformation,
                                  &limits, sizeof(limits), nullptr) == 0) {
        LOG_ERROR(<< "Error querying Job Object information: " << ml::core::CWindowsError().errorString());
        return;
    }

    // Limit the number of active processes to 1 and
    // flag that the limit is set
    limits.ActiveProcessLimit = uint32_t{1};
    limits.LimitFlags = limits.LimitFlags | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
    if (SetInformationJobObject(job, JobObjectBasicLimitInformation,
                                &limits, sizeof(limits)) == 0) {
        LOG_ERROR(<< "Error setting Job information: " << ml::core::CWindowsError().errorString());
        return;
    }

    // Assign current process to the job
    if (AssignProcessToJobObject(job, GetCurrentProcess()) == 0) {
        LOG_ERROR(<< "Error assigning process to Job Object: " << ml::core::CWindowsError().errorString());
        return;
    }

    LOG_DEBUG(<< "ActiveProcessLimit set to 1 for new Job Object");
}
}
}
