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
#include <sandbox/CSandboxedProcessSpawner.h>

#include <core/CLogger.h>

namespace ml {
namespace sandbox {

CSandboxedProcessSpawner::CSandboxedProcessSpawner() = default;

CSandboxedProcessSpawner::~CSandboxedProcessSpawner() = default;

bool CSandboxedProcessSpawner::spawn(const std::string& processPath,
                                     const TStrVec& args,
                                     core::CProcess::TPid& childPid,
                                     std::string* failureReason) {
    if (failureReason != nullptr) {
        *failureReason = "Sandbox2 is not available on this platform";
    }
    LOG_ERROR(<< "Sandbox2 spawn requested for '" << processPath
              << "' but Sandbox2 is not available on this platform");
    return false;
}

bool CSandboxedProcessSpawner::terminateChild(core::CProcess::TPid pid) {
    LOG_WARN(<< "Will not attempt to kill sandboxed process " << pid
             << ": Sandbox2 is not available on this platform");
    return false;
}

bool CSandboxedProcessSpawner::hasChild(core::CProcess::TPid pid) const {
    return false;
}

} // namespace sandbox
} // namespace ml
