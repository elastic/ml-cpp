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

#include <model/CSystemMemoryUsage.h>

#include <core/CProcessStats.h>

namespace ml {
namespace model {
// On Linux the system memory usage is actually that determined by the OS.
// The estimated value provided is ignored.
std::size_t CSystemMemoryUsage::operator()(std::size_t /*memSize*/) const {
    return core::CProcessStats::maxResidentSetSize();
}

std::size_t CSystemMemoryUsage::maybeAdjustUsage(std::size_t usage,
                                                 const TMemoryAdjuster& /*memAdjuster*/) {
    // On Linux we use the actual system memory usage, rather tha an estimated value, so there is no need to adjust it.
    return usage;
}
}
}
