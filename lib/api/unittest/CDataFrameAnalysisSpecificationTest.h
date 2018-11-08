/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CDataFrameAnalysisSpecificationTest_h
#define INCLUDED_CDataFrameAnalysisSpecificationTest_h

#include <cppunit/extensions/HelperMacros.h>

class CDataFrameAnalysisSpecificationTest : public CppUnit::TestFixture {
public:
    void testCreate();
    void testRunAnalysis();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CDataFrameAnalysisSpecificationTest_h
