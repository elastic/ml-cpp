/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#ifndef INCLUDED_CStateCompressorTest_h
#define INCLUDED_CStateCompressorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CStateCompressorTest : public CppUnit::TestFixture {
public:
    void testForApiNoKey();
    void testStreaming();
    void testChunking();
    void testFile();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CStateCompressorTest_h
