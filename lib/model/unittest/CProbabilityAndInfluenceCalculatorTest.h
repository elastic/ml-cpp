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
        void testInfluenceUnavailableCalculator();
        void testLogProbabilityComplementInfluenceCalculator();
        void testMeanInfluenceCalculator();
        void testLogProbabilityInfluenceCalculator();
        void testIndicatorInfluenceCalculator();
        void testProbabilityAndInfluenceCalculator();

        static CppUnit::Test *suite();
};

#endif // INCLUDED_CProbabilityAndInfluenceCalculatorTest_h
