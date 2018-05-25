/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CRandomizedPeriodicityTestTest_h
#define INCLUDED_CRandomizedPeriodicityTestTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRandomizedPeriodicityTestTest : public CppUnit::TestFixture {
public:
    void testAccuracy();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CRandomizedPeriodicityTestTest_h
