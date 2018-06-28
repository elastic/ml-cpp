/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CLogTDistributionTest_h
#define INCLUDED_CLogTDistributionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CLogTDistributionTest : public CppUnit::TestFixture {
public:
    void testMode();
    void testPdf();
    void testCdf();
    void testQuantile();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CLogTDistributionTest_h
