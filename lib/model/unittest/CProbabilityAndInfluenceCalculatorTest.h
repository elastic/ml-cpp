/*
 * ELASTICSEARCH CONFIDENTIAL
 *
 * Copyright (c) 2016 Elasticsearch BV. All Rights Reserved.
 *
 * Notice: this software, and all information contained
 * therein, is the exclusive property of Elasticsearch BV
 * and its licensors, if any, and is protected under applicable
 * domestic and foreign law, and international treaties.
 *
 * Reproduction, republication or distribution without the
 * express written consent of Elasticsearch BV is
 * strictly prohibited.
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
