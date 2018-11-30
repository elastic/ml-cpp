/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CStaticThreadPoolTest_h
#define INCLUDED_CStaticThreadPoolTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStaticThreadPoolTest : public CppUnit::TestFixture {
public:
    void testScheduleDelayMinimisation();
    void testThroughputStability();
    void testManyTasksThroughput();
    void testExceptions();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CStateMachineTest_h
