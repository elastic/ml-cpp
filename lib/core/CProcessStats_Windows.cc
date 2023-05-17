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
#include <core/CProcessMemory.h>

#include <core/WindowsSafe.h>
#include <psapi.h>

#pragma comment(lib, "psapi.lib")

namespace ml {
namespace core {

std::size_t CProcessMemory::residentSetSize() {
    PROCESS_MEMORY_COUNTERS stats;
    GetProcessMemoryInfo(GetCurrentProcess(), &stats, sizeof(info));
    return static_cast<std::size_t>(stats.WorkingSetSize);
}

std::size_t CProcessMemory::maxResidentSetSize() {
    PROCESS_MEMORY_COUNTERS stats;
    GetProcessMemoryInfo(GetCurrentProcess(), &stats, sizeof(info));
    return static_cast<std::size_t>(stats.PeakWorkingSetSize);
}
}
}
