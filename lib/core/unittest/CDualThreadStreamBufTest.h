/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CDualThreadStreamBufTest_h
#define INCLUDED_CDualThreadStreamBufTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDualThreadStreamBufTest : public CppUnit::TestFixture {
public:
    void testThroughput();
    void testSlowConsumer();
    void testPutback();
    void testFatal();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDualThreadStreamBufTest_h
