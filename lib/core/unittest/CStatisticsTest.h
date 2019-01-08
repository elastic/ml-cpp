/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CStatisticsTest_h
#define INCLUDED_CStatisticsTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStatisticsTest : public CppUnit::TestFixture {
public:
    void testStatistics();
    void testPersist();
    void testCacheStatistics();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CStatisticsTest_h
