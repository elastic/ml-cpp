/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CMockSearcher.h"

#include <core/CLogger.h>

#include "CMockDataAdder.h"

CMockSearcher::CMockSearcher(const CMockDataAdder& mockDataAdder) : m_MockDataAdder(mockDataAdder) {
}

CMockSearcher::TIStreamP CMockSearcher::search(size_t currentDocNum, size_t /*limit*/) {
    if (currentDocNum == 0) {
        LOG_ERROR(<< "Current doc number cannot be 0 - data store requires 1-based numbers");
        return TIStreamP();
    }

    TIStreamP stream;
    const CMockDataAdder::TStrStrVecMap events = m_MockDataAdder.events();

    CMockDataAdder::TStrStrVecMapCItr iter = events.find(m_SearchTerms[0]);
    if (iter == events.end()) {
        LOG_TRACE(<< "Can't find search " << m_SearchTerms[0]);
        stream.reset(new std::stringstream("{}"));
    } else {
        LOG_TRACE(<< "Got search data for " << m_SearchTerms[0]);
        if (currentDocNum > iter->second.size()) {
            stream.reset(new std::stringstream("[ ]"));
        } else {
            stream.reset(new std::stringstream(iter->second[currentDocNum - 1]));
        }
    }
    return stream;
}
