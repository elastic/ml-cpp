/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalyzerOutlierTest_h
#define INCLUDED_CDataFrameAnalyzerOutlierTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalyzerOutlierTest : public CppUnit::TestFixture {
public:
    void testWithoutControlMessages();
    void testRunOutlierDetection();
    void testRunOutlierDetectionPartitioned();
    void testRunOutlierFeatureInfluences();
    void testRunOutlierDetectionWithParams();
    void testFlushMessage();
    void testErrors();
    void testRoundTripDocHashes();
    void testCategoricalFields();
    void testCategoricalFieldsEmptyAsMissing();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameAnalyzerOutlierTest_h
