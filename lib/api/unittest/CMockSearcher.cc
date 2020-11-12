/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CMockSearcher.h"

#include <core/CLogger.h>

#include "CMockDataAdder.h"

CMockSearcher::CMockSearcher(const CMockDataAdder& mockDataAdder)
    : m_MockDataAdder{mockDataAdder} {
}

CMockSearcher::TIStreamP CMockSearcher::search(std::size_t currentDocNum, std::size_t /*limit*/) {
    if (currentDocNum == 0) {
        LOG_ERROR(<< "Current doc number cannot be 0 - data store requires 1-based numbers");
        return TIStreamP{};
    }

    const CMockDataAdder::TStrVec& events = m_MockDataAdder.events();
    if (currentDocNum > events.size()) {
        return TIStreamP{new std::istringstream("[ ]")};
    }
    return TIStreamP{new std::istringstream(events[currentDocNum - 1])};
}
