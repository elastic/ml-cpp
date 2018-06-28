/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CMultivariateConstantPriorTest_h
#define INCLUDED_CMultivariateConstantPriorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CMultivariateConstantPriorTest : public CppUnit::TestFixture {
public:
    void testAddSamples();
    void testMarginalLikelihood();
    void testMarginalLikelihoodMean();
    void testMarginalLikelihoodMode();
    void testMarginalLikelihoodCovariance();
    void testSampleMarginalLikelihood();
    void testProbabilityOfLessLikelySamples();
    void testPersist();

    static CppUnit::Test* suite();
};

#endif // INCLUDED_CMultivariateConstantPriorTest_h
