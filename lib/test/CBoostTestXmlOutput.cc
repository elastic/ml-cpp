/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include <test/CBoostTestXmlOutput.h>

#include <boost/test/unit_test_log.hpp>

namespace ml {
namespace test {

std::ofstream CBoostTestXmlOutput::ms_XmlOutputFile("boost_test_results.xml");

bool CBoostTestXmlOutput::init() {
    // Add XML output to boost_test_results.xml as an extra output over and above
    // the default or whatever was specified on the command line
    boost::unit_test::unit_test_log.add_format(boost::unit_test::OF_XML);
    boost::unit_test::unit_test_log.set_stream(boost::unit_test::OF_XML, ms_XmlOutputFile);
    boost::unit_test::unit_test_log.set_threshold_level(
        boost::unit_test::OF_XML, boost::unit_test::log_test_units);
    return true;
}
}
}
