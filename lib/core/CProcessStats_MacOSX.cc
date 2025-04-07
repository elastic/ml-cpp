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

#include <errno.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/time.h>

namespace ml {
namespace core {

std::size_t CProcessStats::residentSetSize() {
    // not supported on osx
    return 0;
}

std::size_t CProcessStats::maxResidentSetSize() {
    struct rusage rusage;

    if (getrusage(RUSAGE_SELF, &rusage) != 0) {
        LOG_DEBUG(<< "failed to get resource usage(getrusage): " << ::strerror(errno));
        return 0;
    }
    auto maxRSS = static_cast<std::size_t>(rusage.ru_maxrss);
    CProgramCounters::counter(counter_t::E_TSADMaxResidentSetSize) = maxRSS;
    // ru_maxrss is in bytes
    return maxRSS;
}
}
}
