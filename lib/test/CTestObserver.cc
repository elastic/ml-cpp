/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CTestObserver.h>

#include <core/CLogger.h>

#include <boost/test/tree/test_unit.hpp>

#include <string>

namespace ml {
namespace test {

void CTestObserver::test_unit_start(const boost::unit_test::test_unit& test) {
    const std::string& unitName{test.full_name()};
    if (unitName.find('/') == std::string::npos) {
        // This gets called for suites as well as test cases - ignore these
        return;
    }
    LOG_DEBUG(<< '+' << std::string(unitName.length() + 4, '-') << '+');
    LOG_DEBUG(<< "|  " << unitName << "  |");
    LOG_DEBUG(<< '+' << std::string(unitName.length() + 4, '-') << '+');
}

void CTestObserver::test_unit_finish(const boost::unit_test::test_unit& test, unsigned long elapsed) {
    const std::string& unitName{test.full_name()};
    if (unitName.find('/') == std::string::npos) {
        // This gets called for suites as well as test cases - ignore these
        return;
    }
    LOG_INFO(<< "Unit test timing - " << unitName << " took " << (elapsed / 1000) << "ms");
}
}
}
