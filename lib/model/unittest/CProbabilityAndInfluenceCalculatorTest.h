/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */

#ifndef INCLUDED_CProbabilityAndInfluenceCalculatorTest_h
#define INCLUDED_CProbabilityAndInfluenceCalculatorTest_h

#include <cppunit/extensions/HelperMacros.h>

class CProbabilityAndInfluenceCalculatorTest : public CppUnit::TestFixture
{
    public:
        void testInfluenceUnavailableCalculator(void);
        void testLogProbabilityComplementInfluenceCalculator(void);
        void testMeanInfluenceCalculator(void);
        void testLogProbabilityInfluenceCalculator(void);
        void testIndicatorInfluenceCalculator(void);
        void testProbabilityAndInfluenceCalculator(void);

        static CppUnit::Test *suite(void);
};

#endif // INCLUDED_CProbabilityAndInfluenceCalculatorTest_h
