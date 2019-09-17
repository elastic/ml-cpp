/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalyzerInferenceOutputTest_h
#define INCLUDED_CDataFrameAnalyzerInferenceOutputTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalyzerInferenceOutputTest : public CppUnit::TestFixture {
public:
    void testTrainOneHotEncoding();
    void testTrainTargetMeanEncoding();
    void testTrainFrequencyEncoding();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameAnalyzerInferenceOutputTest_h
