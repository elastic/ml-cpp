/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CRegressionTest_h
#define INCLUDED_CRegressionTest_h

#include <cppunit/extensions/HelperMacros.h>

class CRegressionTest : public CppUnit::TestFixture
{
    public:
        void testInvariants(void);
        void testFit(void);
        void testShiftAbscissa(void);
        void testShiftOrdinate(void);
        void testShiftGradient(void);
        void testAge(void);
        void testPrediction(void);
        void testCombination(void);
        void testSingular(void);
        void testScale(void);
        void testMean(void);
        void testCovariances(void);
        void testParameters(void);
        void testPersist(void);
        void testParameterProcess(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CRegressionTest_h
