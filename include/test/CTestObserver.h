/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_ml_test_CTestObserver_h
#define INCLUDED_ml_test_CTestObserver_h

#include <test/ImportExport.h>

#include <boost/test/tree/observer.hpp>

namespace ml {
namespace test {

//! \brief
//! A class to observe Boost.Test tests.
//!
//! DESCRIPTION:\n
//! Logs the test name at the beginning of the test and the
//! elapsed time at the end.
//!
//! IMPLEMENTATION DECISIONS:\n
//! Output format matches what the old CTestTimer class did
//! when we used CppUnit.
//!
class TEST_EXPORT CTestObserver : public boost::unit_test::test_observer {
public:
    //! Called at the start of each suite and at the start of each test case.
    void test_unit_start(const boost::unit_test::test_unit& test) override;

    //! Called at the end of each suite and at the end of each test case.
    void test_unit_finish(const boost::unit_test::test_unit& test, unsigned long elapsed) override;
};
}
}

#endif // INCLUDED_ml_test_CTestObserver_h
