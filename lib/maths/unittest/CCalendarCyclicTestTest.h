/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CCalendarCyclicTestTest_h
#define INCLUDED_CCalendarCyclicTestTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCalendarCyclicTestTest : public CppUnit::TestFixture {
public:
    void testTruePositives();
    void testFalsePositives();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CCalendarCyclicTestTest_h
