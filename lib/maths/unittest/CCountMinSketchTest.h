/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CCountMinSketchTest_h
#define INCLUDED_CCountMinSketchTest_h

#include <cppunit/extensions/HelperMacros.h>

class CCountMinSketchTest : public CppUnit::TestFixture {
public:
    void testCounts();
    void testSwap();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CCountMinSketchTest_h
