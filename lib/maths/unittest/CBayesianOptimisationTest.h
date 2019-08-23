/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CBayesianOptimisationTest_h
#define INCLUDED_CBayesianOptimisationTest_h

#include <cppunit/extensions/HelperMacros.h>

class CBayesianOptimisationTest : public CppUnit::TestFixture {
public:
    void testLikelihoodGradient();
    void testMaximumLikelihoodKernel();
    void testExpectedImprovementGradient();
    void testMaximumExpectedImprovement();
    void testPersistRestore();

    static CppUnit::Test* suite();

private:
    void testPersistRestoreIsIdempotent(const std::vector<double>& minBoundary,
                                        const std::vector<double>& maxBoundary,
                                        const std::vector<std::vector<double>>& parameterFunctionValues) const;
};

#endif // INCLUDED_CBayesianOptimisationTest_h
