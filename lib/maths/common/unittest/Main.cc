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

#define BOOST_TEST_MODULE lib.maths.common
// Defining BOOST_TEST_MODULE usually auto-generates main(), but we don't want
// this as we need custom initialisation to allow for output in both console and
// Boost.Test XML formats
#define BOOST_TEST_NO_MAIN

#include <test/CBoostTestXmlOutput.h>
#include <test/CTestObserver.h>

#include <boost/test/unit_test.hpp>

int main(int argc, char** argv) {
    ml::test::CTestObserver observer;
    boost::unit_test::framework::register_observer(observer);
    int result{boost::unit_test::unit_test_main(&ml::test::CBoostTestXmlOutput::init,
                                                argc, argv)};
    boost::unit_test::framework::deregister_observer(observer);
    return result;
}
