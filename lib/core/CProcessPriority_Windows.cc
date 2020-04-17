/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <core/CProcessPriority.h>

namespace ml {
namespace core {

void CProcessPriority::reduceMemoryPriority() {
    // Default is to do nothing - see platform-specific implementation files for
    // platforms where we do more
}

void CProcessPriority::reduceCpuPriority() {
    // Nothing at present on Windows
}
}
}
