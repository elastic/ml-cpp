/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMultivariateMultimodalPriorTest_h
#define INCLUDED_CMultivariateMultimodalPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultivariateMultimodalPriorTest : public CppUnit::TestFixture {
public:
    void testMultipleUpdate();
    void testPropagation();
    void testSingleMode();
    void testMultipleModes();
    void testSplitAndMerge();
    void testMarginalLikelihood();
    void testMarginalLikelihoodMean();
    void testMarginalLikelihoodMode();
    void testSampleMarginalLikelihood();
    void testProbabilityOfLessLikelySamples();
    void testIntegerData();
    void testLowVariationData();
    void testLatLongData();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CMultivariateMultimodalPriorTest_h
