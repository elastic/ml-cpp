/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CBoostedTreeRegressionInferenceModelFormatterTest_h
#define INCLUDED_CBoostedTreeRegressionInferenceModelFormatterTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBoostedTreeRegressionInferenceModelFormatterTest : public CppUnit::TestFixture {
public:
    void testIntegration();
    void testDefinitionGeneration();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CBoostedTreeRegressionInferenceModelFormatterTest_h
