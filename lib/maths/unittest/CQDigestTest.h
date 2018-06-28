/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CQDigestTest_h
#define INCLUDED_CQDigestTest_h

#include <cppunit/extensions/HelperMacros.h>

class CQDigestTest : public CppUnit::TestFixture {
public:
    void testAdd();
    void testMerge();
    void testCdf();
    void testSummary();
    void testPropagateForwardByTime();
    void testScale();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CQDigestTest_h
