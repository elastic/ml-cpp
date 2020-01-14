/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <model/CMonitoredResource.h>

namespace ml {
namespace model {

bool CMonitoredResource::supportsPruning() const {
    return false;
}

bool CMonitoredResource::initPruneWindow(std::size_t& /*defaultPruneWindow*/,
                                         std::size_t& /*minimumPruneWindow*/) const {
    return false;
}

core_t::TTime CMonitoredResource::bucketLength() const {
    return 0;
}

void CMonitoredResource::prune(std::size_t /*maximumAge*/) {
    // NO-OP
}
}
}
