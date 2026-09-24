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
#include <seccomp/CLandlockFilesystemPolicy.h>

// Landlock is a Linux LSM; ml_generate_platform_sources() substitutes
// CLandlockFilesystemPolicy_Linux.cc for this file on Linux builds. This
// translation unit is what macOS and Windows compile instead, so a caller
// can stay platform-independent and simply observe E_Unsupported.

namespace ml {
namespace seccomp {

std::string describe(ELandlockOutcome outcome) {
    switch (outcome) {
    case ELandlockOutcome::E_Applied:
        return "applied";
    case ELandlockOutcome::E_Unsupported:
        return "unsupported on this platform";
    case ELandlockOutcome::E_Failed:
        return "failed";
    }
    return "unrecognized outcome";
}

int landlockAbiVersion() {
    return 0;
}

SLandlockPaths pytorchInferenceLandlockPaths(const std::string& /*ipcDirectory*/) {
    return SLandlockPaths{};
}

ELandlockOutcome applyLandlockFilesystemPolicy(const SLandlockPaths& /*paths*/) {
    return ELandlockOutcome::E_Unsupported;
}

} // namespace seccomp
} // namespace ml
