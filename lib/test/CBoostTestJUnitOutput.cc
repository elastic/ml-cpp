/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CBoostTestJUnitOutput.h>

#include <boost/test/unit_test_log.hpp>

namespace ml {
namespace test {

std::ofstream CBoostTestJUnitOutput::ms_JUnitOutputFile("junit_results.xml");

bool CBoostTestJUnitOutput::init() {
    // Add JUnit output to junit_results.xml as an extra output over and above
    // the default or whatever was specified on the command line
    boost::unit_test::unit_test_log.add_format(boost::unit_test::OF_JUNIT);
    boost::unit_test::unit_test_log.set_stream(boost::unit_test::OF_JUNIT, ms_JUnitOutputFile);
    return true;
}
}
}
