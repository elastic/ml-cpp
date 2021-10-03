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
