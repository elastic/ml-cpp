/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalyzerTest_h
#define INCLUDED_CDataFrameAnalyzerTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalyzerTest : public CppUnit::TestFixture {
public:
    void testWithoutControlMessages();
    void testRunOutlierDetection();
    void testFlushMessage();
    void testErrors();
    void testRoundTripDocIds();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameAnalyzerTest_h
