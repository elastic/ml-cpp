/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CProcessPriority.h>

#include <core/CLogger.h>

#include <errno.h>
#include <string.h>
#include <unistd.h>

namespace ml {
namespace core {

void CProcessPriority::reduceMemoryPriority() {
    // Default is to do nothing - see platform-specific implementation files for
    // platforms where we do more
}

void CProcessPriority::reduceCpuPriority() {
    errno = 0;
    if (::nice(5) == -1 && errno != 0) {
        LOG_ERROR(<< "Failed to reduce process priority: " << ::strerror(errno));
    }
}
}
}
