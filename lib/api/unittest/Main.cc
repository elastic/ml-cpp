/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#define BOOST_TEST_MODULE lib.api
// Defining BOOST_TEST_MODULE usually auto-generates main(), but we don't want
// this as we need custom initialisation to allow for output in both console and
// JUnit formats
#define BOOST_TEST_NO_MAIN

#include <test/CBoostTestJUnitOutput.h>
#include <test/CTestObserver.h>

#include <boost/test/unit_test.hpp>

int main(int argc, char** argv) {
    ml::test::CTestObserver observer;
    boost::unit_test::framework::register_observer(observer);
    return boost::unit_test::unit_test_main(&ml::test::CBoostTestJUnitOutput::init,
                                            argc, argv);
    boost::unit_test::framework::deregister_observer(observer);
}
