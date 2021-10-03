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

#include <test/CProgramCounterClearingFixture.h>

#include <core/CProgramCounters.h>

namespace ml {
namespace test {

CProgramCounterClearingFixture::CProgramCounterClearingFixture() {
    core::CProgramCounters& counters{core::CProgramCounters::instance()};

    // Set all counters to 0
    for (std::size_t i = 0; i < counter_t::NUM_COUNTERS; ++i) {
        counters.counter(i) = 0;
    }

    // Clear the cache
    counters.m_Cache.clear();
}
}
}
