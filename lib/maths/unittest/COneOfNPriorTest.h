/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_COneOfNPriorTest_h
#define INCLUDED_COneOfNPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class COneOfNPriorTest : public CppUnit::TestFixture {
public:
    void testFilter();
    void testMultipleUpdate();
    void testWeights();
    void testModels();
    void testModelSelection();
    void testMarginalLikelihood();
    void testMarginalLikelihoodMean();
    void testMarginalLikelihoodMode();
    void testMarginalLikelihoodVariance();
    void testSampleMarginalLikelihood();
    void testCdf();
    void testProbabilityOfLessLikelySamples();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_COneOfNPriorTest_h
