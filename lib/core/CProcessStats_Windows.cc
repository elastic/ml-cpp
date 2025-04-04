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
#include <core/CProcessStats.h>

#include <core/CLogger.h>
#include <core/CProgramCounters.h>
#include <core/CWindowsError.h>

#include <core/WindowsSafe.h>
#include <psapi.h>

#pragma comment(lib, "psapi.lib")

namespace ml {
namespace core {

std::size_t CProcessStats::residentSetSize() {
    PROCESS_MEMORY_COUNTERS stats;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &stats, sizeof(stats)) == FALSE) {
        LOG_DEBUG(<< "Failed to retrieve memory info " << CWindowsError());
        return 0;
    }

    return static_cast<std::size_t>(stats.WorkingSetSize);
}

std::size_t CProcessStats::maxResidentSetSize() {
    PROCESS_MEMORY_COUNTERS stats;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &stats, sizeof(stats)) == FALSE) {
        LOG_DEBUG(<< "Failed to retrieve memory info " << CWindowsError());
        return 0;
    }

    std::size_t peakWorkingSetSize = static_cast<std::size_t>(stats.PeakWorkingSetSize);

    CProgramCounters::counter(counter_t::E_TSADMaxResidentSetSize) = peakWorkingSetSize;

    return peakWorkingSetSize;
}
}
}
